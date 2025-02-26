import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from pathlib import Path
import re
from nltk import edit_distance
from tqdm import tqdm 

from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel

from donut_dataset import DonutDataset

import datetime

MODEL_TOKEN_START = "<ocr_pck>"
MODEL_TOKEN_END = '<ocr_pck/>'

def compute_edit_distance(pred, answer):
    pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    answer = re.sub(r"<.*?>", "", answer, count=1).replace("</s>", "")
    return edit_distance(pred, answer) / max(len(pred), len(answer))


def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    donut_config = VisionEncoderDecoderConfig.from_pretrained(config["pretrained_model_name_or_path"])
    donut_config.encoder.image_size = config["input_size"]
    donut_config.decoder.max_length = config["max_length"]

    processor = DonutProcessor.from_pretrained(config["pretrained_model_name_or_path"])
    model = VisionEncoderDecoderModel.from_pretrained(config["pretrained_model_name_or_path"], config=donut_config).to(device)

    processor.image_processor.size = config["input_size"][::-1]
    processor.image_processor.do_align_long_axis = False

    datasets = {}
    
    task_name = config["task_name"]

    print("I", datetime.datetime.now().time())

    for split in ["train", "validation"]:

        print("I - a", datetime.datetime.now().time())
        datasets[split] = DonutDataset(
            model,
            processor,
            config["dataset_name_or_path"] + '/' + split,
            max_length=config["max_length"],
            task_start_token=MODEL_TOKEN_START,
            prompt_end_token=MODEL_TOKEN_END,
            sort_json_key=False
        )
    
    print("II", datetime.datetime.now().time())

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([MODEL_TOKEN_START])[0]

    train_loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["validation"], batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    result_path = Path(config["result_path"])
    result_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print("III", datetime.datetime.now().time())
        
    for epoch in range(config["max_epochs"]):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        xd = 0
        for pixel_values, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} - Training"):
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = model(pixel_values, labels=labels).loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            xd += 1
            if xd == 10:
                break # TESTING LOOP
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {train_loss:.4f}")

        print("IV", datetime.datetime.now().time())
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_scores = []
        xd = 0
        with torch.no_grad():
            for pixel_values, labels, answers in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} - Validation"):
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                batch_size = pixel_values.shape[0]

                decoder_input_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)

                outputs = model.generate(pixel_values,
                                        decoder_input_ids=decoder_input_ids,
                                        max_length=model.decoder.config.max_position_embeddings,
                                        pad_token_id=processor.tokenizer.pad_token_id,
                                        eos_token_id=processor.tokenizer.eos_token_id,
                                        use_cache=True,
                                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                        return_dict_in_generate=True,)
            
                preds = []
                for seq in processor.tokenizer.batch_decode(outputs.sequences):
                    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
                    preds.append(seq)

                scores = [compute_edit_distance(pred, ans) for pred, ans in zip(preds, answers[0])]
                val_scores.extend(scores)
                val_loss += sum(scores)

                xd += 1
                if xd == 10:
                    break # TESTING LOOP

        print("V", datetime.datetime.now().time())
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Val Loss: {val_loss:.4f}")

        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(result_path)
            print(f"Model saved to {result_path}")

    print("Training complete!")



config = {
    "max_epochs": 2,
    "k_folds": 5,
    "lr": 1e-4,
    "batch_size": 1,
    "max_length": 768,
    "pretrained_model_name_or_path": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "result_path": "result",
    "seed": 42,
    "sort_json_key": True,
    "dataset_name_or_path": "dataset",
    "task_name": "ocr-pck",
    "input_size": [1280, 960]
}

if __name__ == "__main__":
    train(config)

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from pathlib import Path
import re
from nltk import edit_distance
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel

from donut_dataset import DonutDataset

import datetime
import uuid

MODEL_TOKEN_START = "<ocr_pck>"
MODEL_TOKEN_END = '<ocr_pck/>'

def compute_edit_distance(pred, answer):
    pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    answer = re.sub(r"<.*?>", "", answer, count=1).replace("</s>", "")
    return edit_distance(pred, answer) / max(len(pred), len(answer))

def log_hparams(writer, config, train_loss, val_loss, val_cer, loop_idx, experiment_id):
    """Logging hyperparameters and results to TensorBoard."""
    
    hparams = {
        'experiment_id': experiment_id,
        'batch_size': config["batch_size"],
        'lr': config["lr"],
        'max_epochs': config["max_epochs"],
        'max_length': config["max_length"]
    }
    
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_cer': val_cer
    }
    
    # Log hparams and metrics
    writer.add_hparams(hparams, metrics, global_step=loop_idx)


def train(config):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

    # Initialize TensorBoard writer
    log_dir = result_path / "logs"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Generate unique experiment ID
    experiment_id = str(uuid.uuid4())

    best_val_loss = float("inf")
    early_stopping_patience = config["early_stopping_patience"]
    early_stopping_counter = 0

    print("III", datetime.datetime.now().time())
        
    for epoch in range(config["max_epochs"]):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        for pixel_values, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} - Training"):
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = model(pixel_values, labels=labels).loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {train_loss:.4f}")

        # Log training loss to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)

        print("IV", datetime.datetime.now().time())
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_scores = []
        with torch.no_grad():
            for pixel_values, labels, answers in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} - Validation"):
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                batch_size = pixel_values.shape[0]

                decoder_input_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)

                outputs = model.generate(pixel_values,
                                        decoder_input_ids=decoder_input_ids,
                                        max_length=config["max_length"],
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

                if len(preds) != len(answers[0]):
                    print(f"Warning: Number of predictions ({len(preds)}) does not match number of answers ({len(answers[0])})")
                    continue

                scores = [compute_edit_distance(pred, ans) for pred, ans in zip(preds, answers[0])]
                val_scores.extend(scores)
                val_loss += sum(scores)

        print("V", datetime.datetime.now().time())
        val_loss /= len(val_loader)
        val_cer = np.mean(val_scores)
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}")

        # Log validation loss and CER to TensorBoard
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("CER/Validation", val_cer, epoch)

        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(result_path)
            print(f"Model saved to {result_path}")
            early_stopping_counter = 0  # reset the counter if we get a new best model
            log_hparams(writer, config, train_loss, val_loss, val_cer, epoch, experiment_id)
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    writer.close()
    print("Training complete!")


config = {
    "max_epochs": 20,
    "lr": 1e-4,
    "batch_size": 2,
    "max_length": 256,
    "pretrained_model_name_or_path": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "result_path": "result",
    "dataset_name_or_path": "dataset",
    "task_name": "ocr-pck",
    "input_size": [1280, 960],
    "early_stopping_patience": 3
}

if __name__ == "__main__":
    train(config)
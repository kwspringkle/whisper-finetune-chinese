import re
import os
import gc
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from multiprocessing import Pool
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_callback import EarlyStoppingCallback
from .utils import(
    clean_text,
    is_valid,
    apply_local_timestamps,
    preprocess_dataset,
    config_tokenizer,
    delete_all_checkpoints
)

from .custom import(
    MemoryCleanupCallback,
    decode_with_timestamps,
    CustomEvalCallback,
    CustomSeq2SeqTrainer
)

# --- Config môi trường ---

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = os.getenv("WANDB_DISABLED", "true")

# --- Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int = None
    max_label_length: int = 448

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Đệm input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Đệm nhãn với padding bên phải (right padding)
        label_features = [{"input_ids": feature["labels"][:self.max_label_length]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            return_tensors="pt",
            max_length=self.max_label_length, 
            padding="max_length",  
            pad_to_multiple_of=None
        )

        labels = labels_batch["input_ids"].clone()
        attention_mask = labels_batch["attention_mask"]

        # Xác định timestamp tokens
        timestamp_token_ids = []
        for token, token_id in self.processor.tokenizer.get_vocab().items():
            if token.startswith("<|") and token.endswith("|>") and any(c.isdigit() for c in token):
                timestamp_token_ids.append(int(token_id))

        # Giữ tất cả non-padding tokens
        keep_mask = attention_mask.bool()
        
        # Áp dụng mask
        labels = labels.masked_fill(~keep_mask, -100)
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        
        return batch

# --- Tính toán Metrics ---
def compute_metrics(pred):
    """
    Tính toán Tỷ lệ lỗi ký tự (CER).
    """
    if isinstance(pred.predictions, tuple):
        print("Tuple")
        pred_ids = np.argmax(pred.predictions[0], axis=-1)
    else:
        print("Not tuple")
        pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = decode_with_timestamps(processor, pred_ids, skip_special_tokens=True)
    label_str = decode_with_timestamps(processor, label_ids, skip_special_tokens=True)

    pred_clean = [clean_text(s) for s in pred_str]
    label_clean = [clean_text(s) for s in label_str]

    for i in range(min(3, len(pred_str))):
        print(f"\n=== Sample {i+1} ===")
        print(f"Pred ids:   {pred_ids[i]}")
        print(f"Label ids:  {label_ids[i]}")
        print(f"Pred raw:   {pred_str[i]}")
        print(f"Label raw:  {label_str[i]}")
        print(f"Pred clean: {pred_clean[i]}")
        print(f"Label clean:{label_clean[i]}")
        print("-" * 50)

    cer = metric.compute(predictions=pred_clean, references=label_clean) * 100
    return {"cer": cer,
           "predictions": pred_str,
            "labels": label_str}

# --- Hàm huấn luyện chính ---         
def train(
    model,
    data_collator,
    processor,
    train_dataset,
    eval_dataset=None,
    output_dir="./whisper-chinese-timestamp",
    num_epoch=5,
    batch_size=4,
    learning_rate=1e-5,
    metric=evaluate.load("cer"),
):
    """
    Hàm huấn luyện chính cho mô hình Whisper.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        eval_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_epoch,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="cer" if eval_dataset else None,
        greater_is_better=False,
        fp16=True,
        fp16_full_eval=True,
        save_total_limit=1,
        lr_scheduler_type="reduce_lr_on_plateau",
        warmup_ratio=0.05,
        push_to_hub=True,
        hub_model_id="kwspringkles/whisper-medium-finetuned",
        max_grad_norm=1.0,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=445,
        generation_num_beams=1,
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3), 
            MemoryCleanupCallback()],
    )
    trainer.add_callback(CustomEvalCallback(trainer=trainer, processor=processor, metric=metric))

    print("Start training Whisper with timestamps...")
    print(f"Total training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Total eval samples: {len(eval_dataset)}")
    trainer._ensure_integer_token_ids(model)
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)

    delete_all_checkpoints(output_dir)
    print(f"\n✅ Model saved to: {output_dir}")
    trainer.push_to_hub()
    return trainer
        
# ===================================================================
# MAIN
# ===================================================================

# --- Khởi tạo Metric và Login ---

metric = evaluate.load("cer")
print("Logging in to Hugging Face...")
login(token=os.getenv("hf_write_token"))

model_name = "openai/whisper-medium"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Khởi tạo Model và Processor ---

print("Initializing model...")
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(
    model_name, language="chinese", task="transcribe"
)

# --- Cấu hình Model ---

print("Configuring model...")
model.config.dropout = 0.2
model.config.attention_dropout = 0.2
model.config.activation_dropout = 0.2

# Đảm bảo tất cả config IDs là integers
model.config.pad_token_id = int(model.config.pad_token_id or 50257)
model.config.eos_token_id = int(model.config.eos_token_id or 50257)
model.config.decoder_start_token_id = int(model.config.decoder_start_token_id or 50257)

forced_decoder_ids = [
    (1, 50260),  # Start of transcript
    (2, 50359),  # Chinese language token
]
model.generation_config.forced_decoder_ids = forced_decoder_ids

model.generation_config.return_timestamps = True
model.generation_config.language = "zh"
model.generation_config.task = "transcribe"
# Đảm bảo tất cả token IDs trong generation config là integers
model.generation_config.pad_token_id = int(model.config.pad_token_id)
model.generation_config.eos_token_id = int(model.config.eos_token_id)
model.generation_config.decoder_start_token_id = int(model.config.decoder_start_token_id)
# Xử lý suppress_tokens
if hasattr(model.generation_config, 'suppress_tokens'):
    suppress_tokens = model.generation_config.suppress_tokens or []
    model.generation_config.suppress_tokens = []

print("=== DEBUG TOKEN IDS ===")
print("Model config eos_token_id:", model.config.eos_token_id, type(model.config.eos_token_id))
print("Model config pad_token_id:", model.config.pad_token_id, type(model.config.pad_token_id))
print("Generation config eos_token_id:", model.generation_config.eos_token_id, type(model.generation_config.eos_token_id))
print("Generation config pad_token_id:", model.generation_config.pad_token_id, type(model.generation_config.pad_token_id))
print("Forced decoder IDs:", forced_decoder_ids)

# --- Thêm Token mới ---

print("Adding new pad token to vocabulary...")
new_pad_token = "<|pad|>"
processor.tokenizer.add_tokens([new_pad_token])
new_pad_token_id = processor.tokenizer.convert_tokens_to_ids(new_pad_token)

# Resize model embeddings và cập nhật pad_token_id
model.resize_token_embeddings(len(processor.tokenizer))
model.config.pad_token_id = new_pad_token_id
model.generation_config.pad_token_id = new_pad_token_id

print(f"New pad token: '{new_pad_token}' -> ID: {new_pad_token_id}")
print(model.generation_config.pad_token_id)

print("Configuring tokenizer...")
processor.tokenizer = config_tokenizer(processor.tokenizer)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# --- Tải và xử lý Dataset ---

print("Loading dataset...")
dataset = load_dataset("kwspringkles/film_final")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=True))
dataset = dataset.filter(is_valid, num_proc=16)

os.makedirs("./whisper-chinese-timestamp", exist_ok=True)
os.makedirs("./results", exist_ok=True)
print("Preprocessing dataset...")
dataset = dataset.map(apply_local_timestamps, num_proc=16)

train_ds = dataset["train"]
val_ds = dataset.get("validation", None)
test_ds = dataset.get("test", None)

train_dataset = preprocess_dataset(train_ds, processor, batch_size=64)
val_dataset = preprocess_dataset(val_ds, processor, batch_size=64)
test_dataset = preprocess_dataset(test_ds, processor, batch_size=64)


# Clean up memory
del dataset
# del train_ds
# del val_ds
gc.collect()
torch.cuda.empty_cache()

# --- Bắt đầu Huấn luyện ---

trainer = train(
    model,
    data_collator,
    processor,
    train_dataset,
    val_dataset,
    output_dir="./whisper-chinese-timestamp",
    num_epoch=15,
    batch_size=8,
    learning_rate=3e-6,
    metric=metric,
)
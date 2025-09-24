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
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
)
from tqdm import tqdm
from transformers.trainer_callback import EarlyStoppingCallback
from utils import(
    clean_text,
    is_valid,
    apply_local_timestamps,
    config_tokenizer,
    backup_and_delete_all_checkpoints,
)
from custom import(
    MemoryCleanupCallback,
    decode_with_timestamps,
    CustomEvalCallback,
    CustomSeq2SeqTrainer
)

from lora_config import wrap_with_lora

def simple_model_check(model, base_model_name="openai/whisper-medium", check_name="Model"):
    """
    Simple check để xem model có thay đổi so với base không
    """
    try:
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        
        # Check 1 layer quan trọng
        layer_name = "model.encoder.layers.0.self_attn.q_proj.weight"
        
        model_weight = dict(model.named_parameters())[layer_name]
        base_weight = dict(base_model.named_parameters())[layer_name]
        
        diff = torch.abs(model_weight - base_weight)
        max_diff = torch.max(diff).item()
        
        print(f"🔍 {check_name}: max diff = {max_diff:.2e}", end="")
        
        if max_diff > 1e-6:
            print(" ✅ CHANGED")
            return True
        else:
            print(" ❌ SAME as base")
            return False
            
    except Exception as e:
        print(f"🔍 {check_name}: ❌ Error - {e}")
        return False
    finally:
        if 'base_model' in locals():
            del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            return_tensors="pt",
        )
        
        # Áp dụng mask
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

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

    pred_str = decode_with_timestamps(processor,pred_ids, skip_special_tokens=True)
    label_str = decode_with_timestamps(processor,label_ids, skip_special_tokens=True)

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
    return {"cer": cer}

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
    base_model_name="openai/whisper-medium",  # Thêm parameter này
):
    """
    Hàm huấn luyện chính cho mô hình Whisper.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        learning_rate=learning_rate,
        num_train_epochs=num_epoch,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=50,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="cer" if eval_dataset else None,
        greater_is_better=False,
        fp16=True,
        fp16_full_eval=True,
        save_total_limit=1,
        lr_scheduler_type="linear",  
        warmup_ratio=0.1,  
        push_to_hub=True,
        hub_model_id="kwspringkles/whisper-medium-new",
        max_grad_norm=1.0, 
        weight_decay=0.01,  
        predict_with_generate=True,
        generation_max_length=445,
        generation_num_beams=1,
        hub_strategy="end",
        dataloader_pin_memory=True,
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
        EarlyStoppingCallback(early_stopping_patience=3),  # Giảm từ 3 xuống 2
        MemoryCleanupCallback(),
    ],
    )
    trainer.add_callback(CustomEvalCallback(trainer=trainer, processor=processor, metric=metric))

    print("Start training Whisper with timestamps...")
    print(f"Total training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Total eval samples: {len(eval_dataset)}")
    trainer._ensure_integer_token_ids(model)
    trainer.train()
    
    # Merge LoRA weights vào base model để tạo model hoàn chỉnh
    print("� Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    # Lưu model hoàn chỉnh (như Whisper bình thường)
    print("💾 Saving merged model...")
    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Backup LoRA adapter riêng (optional)
    lora_backup_dir = output_dir + "_lora_adapter"
    print(f"💾 Backing up LoRA adapter to: {lora_backup_dir}")
    model.save_pretrained(lora_backup_dir)  # Chỉ LoRA weights
    
    # backup_and_delete_all_checkpoints(output_dir, "./backup")
    print(f"\n✅ Complete Whisper model saved to: {output_dir}")
    print(f"✅ LoRA adapter backup saved to: {lora_backup_dir}")
    print("📁 Model can be loaded with: WhisperForConditionalGeneration.from_pretrained()")
    
    # Simple verification trước khi push
    print("\n� Pre-push verification:")
    test_model = WhisperForConditionalGeneration.from_pretrained(output_dir)
    simple_model_check(test_model, base_model_name, "Final Model")
    del test_model
    
    # Push model hoàn chỉnh lên Hub (như Whisper bình thường)
    try:
        print("🚀 Pushing complete Whisper model to Hub...")
        merged_model.push_to_hub("kwspringkles/whisper-medium-new")
        processor.push_to_hub("kwspringkles/whisper-medium-new")
        print("✅ Complete Whisper model pushed to Hub successfully!")
        print("📱 Can be used with: pipeline('automatic-speech-recognition', model='your-username/whisper-medium-chinese-finetune')")
    except Exception as e:
        print(f"⚠️ Failed to push to Hub: {e}")
    
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

# Load base model trước
base_model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Apply LoRA
print("🔄 Applying LoRA configuration...")
model = wrap_with_lora(base_model)

# Simple check
print("� Initial model vs base:", end=" ")
simple_model_check(model, model_name, "Initial LoRA")

model = model.to(device)
processor = WhisperProcessor.from_pretrained(
    model_name, language="chinese", task="transcribe"
)

# --- Cấu hình Model ---

print("Configuring model...")
model.config.dropout = 0.0

# Đảm bảo tất cả config IDs là integers
model.config.pad_token_id = int(model.config.pad_token_id or 50257)
model.config.eos_token_id = int(model.config.eos_token_id or 50257)
model.config.decoder_start_token_id = int(model.config.decoder_start_token_id or 50257)
model.generation_config.return_timestamps = True
model.generation_config.language = "zh"
model.generation_config.task = "transcribe"
forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe", no_timestamps=False)
model.config.forced_decoder_ids = forced_decoder_ids
# Đảm bảo tất cả token IDs trong generation config là integers
model.generation_config.pad_token_id = int(model.config.pad_token_id)
model.generation_config.eos_token_id = int(model.config.eos_token_id)
model.generation_config.decoder_start_token_id = int(model.config.decoder_start_token_id)
# # Xử lý suppress_tokens
# model.config.suppress_tokens = []
# model.generation_config.suppress_tokens = []

print("=== DEBUG TOKEN IDS ===")
print("Model config eos_token_id:", model.config.eos_token_id, type(model.config.eos_token_id))
print("Model config pad_token_id:", model.config.pad_token_id, type(model.config.pad_token_id))
print("Generation config eos_token_id:", model.generation_config.eos_token_id, type(model.generation_config.eos_token_id))
print("Generation config pad_token_id:", model.generation_config.pad_token_id, type(model.generation_config.pad_token_id))
print("Forced decoder IDs:", forced_decoder_ids)
model.generation_config.pad_token_id = model.config.pad_token_id
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

print("Preprocessing dataset...")
dataset = dataset.map(apply_local_timestamps, num_proc=16)
os.makedirs("./whisper-chinese-timestamp", exist_ok=True)
os.makedirs("./results", exist_ok=True)


def prepare_dataset(batch):
    audio = batch["audio"]
    text = batch["chinese_local"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(text,truncation=True,padding=False, max_length=448).input_ids
    return {"input_features": batch["input_features"], "labels": batch["labels"]}

dataset = dataset.map(
    prepare_dataset,
    num_proc=16,  
    load_from_cache_file=True,  
    desc="Preprocessing audio and text" 
)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]
# Clean up memory
del dataset
# del train_ds
# del val_ds
gc.collect()
torch.cuda.empty_cache()


# --- Kiểm tra Data Collator với mẫu dữ liệu thực ---

print("\n=== Kiểm tra Data Collator ===")

# Lấy 3 mẫu đầu từ train_dataset
sample_features = [train_dataset[i] for i in range(min(3, len(train_dataset)))]



# Áp dụng data collator
batch = data_collator(sample_features)


print("=== Chi tiết Labels (một phần) ===")
print(f"Labels tensor (first 10 tokens per sample):\n{batch['labels']}")

# --- Bắt đầu Huấn luyện ---

trainer = train(
    model,
    data_collator,
    processor,
    train_dataset,
    val_dataset,
    output_dir="./whisper-chinese-timestamp",
    num_epoch=10,
    batch_size=4,
    learning_rate=1e-5,  
    metric=metric,
    base_model_name=model_name,  # Thêm parameter này
)
import re
import os
import gc
import json
from typing import List, Union
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
)
from utils import clean_text

# --- Hàm giải mã tùy chỉnh --- 
def decode_with_timestamps(
    processor, pred_ids: Union[torch.Tensor, np.ndarray], skip_special_tokens=True
) -> List[str]:
    """
    Hàm giải mã tùy chỉnh giữ lại token timestamp.
    """
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.cpu().numpy()
    pred_ids = np.asarray(pred_ids)
    if pred_ids.ndim == 1:
        pred_ids = np.expand_dims(pred_ids, 0)

    tokenizer = processor.tokenizer
    decoded_texts = []
    skip_set = {"<|startoftranscript|>", "<|endoftext|>", "<|zh|>", "<|transcribe|>", "<|pad|>"}

    for seq in pred_ids:
        ids = [int(x) for x in seq if int(x) != -100]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        pieces = []
        normal_acc = []

        for tok in tokens:
            if tok.startswith("<|") and tok.endswith("|>") and any(c.isdigit() for c in tok):
                if normal_acc:
                    pieces.append(tokenizer.convert_tokens_to_string(normal_acc))
                    normal_acc = []
                pieces.append(tok)
                continue

            if tok in skip_set:
                if skip_special_tokens:
                    if normal_acc:
                        pieces.append(tokenizer.convert_tokens_to_string(normal_acc))
                        normal_acc = []
                    continue
                else:
                    if normal_acc:
                        pieces.append(tokenizer.convert_tokens_to_string(normal_acc))
                        normal_acc = []
                    pieces.append(tok)
                    continue

            normal_acc.append(tok)

        if normal_acc:
            pieces.append(tokenizer.convert_tokens_to_string(normal_acc))
        decoded_texts.append("".join(pieces))

    return decoded_texts

# --- Callback để dọn dẹp bộ nhớ ---
class MemoryCleanupCallback(TrainerCallback):
    """
    Callback để dọn dẹp bộ nhớ sau mỗi bước đánh giá.
    """
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

# --- Custom Evaluation Callback ---
class CustomEvalCallback(TrainerCallback):
    """
    Callback để ghi log CER chi tiết trên tập huấn luyện và validation ở cuối mỗi epoch,
    và lưu kết quả chi tiết vào file Excel và JSON.
    """
    
    def __init__(self, trainer, processor, metric, output_dir="./results"):
        self.trainer = trainer
        self.processor = processor
        self.metric = metric
        self.output_dir = output_dir
        
        # File paths
        self.cer_history_path = os.path.join(output_dir, "cer_history.json")
        self.train_excel_path = os.path.join(output_dir, "train_predictions.xlsx")
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CER history file
        if not os.path.exists(self.cer_history_path):
            with open(self.cer_history_path, "w") as f:
                json.dump([], f)

    def split_text_by_timestamps(self, text):
        """
        Tách đoạn text theo timestamp, lọc kí tự đặc biệt nhưng giữ lại một số kí tự có ích  
        Args:
            text: Văn bản đầu vào với timestamp như "<|1.23|>text content<|4.56|>more content"  
        Returns:
            str: Văn bản đã làm sạch với dòng mới phân tách các đoạn timestamp
        """
        # Handle edge cases
        if not text:
            return ""
        
        # Convert list to string if needed
        if isinstance(text, list):
            if not text:
                return ""
            text = text[0] if text[0] else ""
        
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Improved regex pattern to match timestamps more precisely
        timestamp_pattern = r"<\|(?:\d+(?:\.\d+)?)\|>"
        
        # Split by timestamps and keep the text between them
        parts = re.split(timestamp_pattern, text)
        
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            if part:
                cleaned_part = re.sub(r'[^\u4e00-\u9fff0-9]', '', part)
                
                # Normalize whitespace
                cleaned_part = re.sub(r'\s+', ' ', cleaned_part)
                cleaned_part = cleaned_part.strip()
                
                if cleaned_part:
                    cleaned_parts.append(cleaned_part)
        
        # Join with newlines for better readability in Excel
        return "\n".join(cleaned_parts)

    def run_detailed_evaluation(self, dataset, dataset_name, epoch):
        """
        Generate cho dataset sau mỗi epochepoch
        """
        tqdm.write(f"🔍 Running detailed evaluation on {dataset_name} dataset...")
        
        try:
            
            # Run prediction
            tqdm.write(f"   Predicting on {len(dataset)} samples...")
            predictions = self.trainer.predict(dataset)
            
            # Handle prediction output format
            if isinstance(predictions.predictions, tuple):
                
                logits = predictions.predictions[0]
                pred_ids = np.argmax(logits, axis=-1)
                tqdm.write(f"   Predictions format: tuple, shape: {logits.shape}")
            else:
                logits = predictions.predictions
                if logits.ndim == 3:  # Has logits dimension
                    pred_ids = np.argmax(logits, axis=-1)
                    tqdm.write(f"   Predictions format: logits, shape: {logits.shape}")
                else:  # Already token IDs
                    pred_ids = logits
                    tqdm.write(f"   Predictions format: token IDs, shape: {logits.shape}")
            
            # Process labels
            label_ids = predictions.label_ids.copy()
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            
            # Decode predictions and labels using custom decoder
            tqdm.write("   Decoding predictions and labels...")
            pred_str = decode_with_timestamps(self.processor, pred_ids, skip_special_tokens=True)
            label_str = decode_with_timestamps(self.processor, label_ids, skip_special_tokens=True)
            
            # Clean text for CER calculation
            pred_clean = [clean_text(s) for s in pred_str]
            label_clean = [clean_text(s) for s in label_str]
            
            # Calculate overall CER
            tqdm.write("   Computing CER metrics...")
            overall_cer = self.metric.compute(predictions=pred_clean, references=label_clean) * 100
            
            # Prepare detailed results for each sample
            tqdm.write("   Preparing detailed results...")
            results = []
            
            for i in range(len(pred_str)):
                # Get sample data with safe access
                try:
                    sample = dataset[i] if i < len(dataset) else {}
                except Exception:
                    sample = {}
                
                predicted_with_timestamps = pred_str[i]
                ground_truth_with_timestamps = label_str[i]
                
                # Split into clean lines using the improved function
                pred_lines = self.split_text_by_timestamps(predicted_with_timestamps)
                gt_lines = self.split_text_by_timestamps(ground_truth_with_timestamps)
                
                # Calculate individual CER for this sample
                try:
                    individual_cer = self.metric.compute(
                        predictions=[pred_clean[i]], 
                        references=[label_clean[i]]
                    ) * 100
                except Exception:
                    individual_cer = 0.0
                
                results.append({
                    "index": i,
                    "movie_name": sample.get("movie_name", f"sample_{i}"),
                    "chunk_id": sample.get("chunk_id", f"chunk_{i}"),
                    "start_time": sample.get("start_time", 0.0),
                    "ground_truth_with_timestamps": ground_truth_with_timestamps,
                    "predict_with_timestamps": predicted_with_timestamps,
                    "ground_truth_clean": gt_lines,
                    "predict_clean": pred_lines,
                    "individual_cer(%)": round(individual_cer, 2),
                })
                
                # Log first few samples for debugging
                if i < 3:
                    tqdm.write(f"   Sample {i+1}:")
                    tqdm.write(f"     GT: {ground_truth_with_timestamps[:200]}...")
                    tqdm.write(f"     Pred: {predicted_with_timestamps[:200]}...")
                    tqdm.write(f"     CER: {individual_cer:.2f}%")
            
            # Save to Excel
            self._save_to_excel(results, dataset_name, epoch)
            
            tqdm.write(f"✅ {dataset_name.title()} evaluation completed!")
            tqdm.write(f"📊 Overall CER: {overall_cer:.2f}%")
            tqdm.write(f"📝 Saved {len(results)} samples to Excel")
            
            return overall_cer, results
            
        except Exception as e:
            tqdm.write(f"❌ Error in {dataset_name} evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None, []

    def _save_to_excel(self, results, dataset_name, epoch):
        """Lưu kết quả vào file Excel với xử lý lỗi phù hợp"""
        try:
            df = pd.DataFrame(results)
            excel_path = self.train_excel_path
            sheet_name = f"Epoch_{epoch}_{dataset_name.title()}"
            
            # Ensure Excel directory exists
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            
            # Save to Excel with proper error handling
            try:
                # Try to append to existing file
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            except FileNotFoundError:
                # Create new file if doesn't exist
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                # Fallback: save with timestamp suffix
                fallback_path = excel_path.replace('.xlsx', f'_epoch{epoch}.xlsx')
                df.to_excel(fallback_path, sheet_name=sheet_name, index=False)
                tqdm.write(f"⚠️  Saved to fallback file: {fallback_path}")
                
        except Exception as e:
            tqdm.write(f"❌ Error saving Excel file: {e}")

    def save_cer_history(self, epoch, train_cer):
        """Lưu lịch sử CER vào file JSON với timestamp"""
        try:
            # Load existing history
            if os.path.exists(self.cer_history_path):
                with open(self.cer_history_path, "r") as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add new epoch results
            epoch_results = {
                "epoch": int(epoch),
                "train_cer": train_cer,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            history.append(epoch_results)
            
            # Save updated history
            with open(self.cer_history_path, "w") as f:
                json.dump(history, f, indent=4)
            
            tqdm.write(f"💾 CER history saved: Train={train_cer:.2f}%")
            
        except Exception as e:
            tqdm.write(f"❌ Error saving CER history: {e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Hàm callback chính được gọi ở cuối mỗi epoch
        Chạy đánh giá chi tiết trên cả tập huấn luyện và validation
        """
        tqdm.write("\n" + "="*70)
        tqdm.write(f"🎯 EPOCH {int(state.epoch)}")
        tqdm.write("="*70)
        
        epoch = int(state.epoch)
        
        # Initialize results
        train_cer = None
        
        # 1. Evaluate on Training Set (subset)
        if self.trainer.train_dataset is not None and len(self.trainer.train_dataset) > 0:
            tqdm.write("\n🔍 TRAINING SET EVALUATION")
            tqdm.write("-" * 40)
            
            subset_size = min(len(self.trainer.train_dataset), 100)
            tqdm.write(f"   Using subset of {subset_size} training samples")
            
            train_subset = self.trainer.train_dataset.select(range(subset_size))
            # train_subset = self.trainer.train_dataset
            train_cer, train_results = self.run_detailed_evaluation(
                train_subset, 
                "training", 
                epoch
            )
            
            if train_cer is not None:
                self.trainer.log({
                    "detailed_train_cer": train_cer,
                    "detailed_train_samples": len(train_results)
                })
        else:
            tqdm.write("⚠️  No training dataset available")
        
        # 2. Save CER history to JSON
        tqdm.write("\n💾 SAVING RESULTS")
        tqdm.write("-" * 20)
        self.save_cer_history(epoch, train_cer)
        
        # 3. Summary
        tqdm.write(f"\n📈 EPOCH {epoch} SUMMARY")
        tqdm.write("-" * 30)
        if train_cer is not None:
            tqdm.write(f"   🚂 Training CER:   {train_cer:.2f}%")
        else:
            tqdm.write("   🚂 Training CER:   N/A")
            
        tqdm.write(f"   📁 Results saved to: {self.output_dir}")
        tqdm.write("="*70 + "\n")

        # 5. Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        tqdm.write("🧹 Memory cleanup completed")

class CustomSeq2SeqTrainer(Seq2SeqTrainer):   
    # def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
    #     # Tính loss gốc
    #     self.model_accepts_loss_kwargs=False
    #     outputs = model(**inputs)
    #     loss = outputs.loss

    #     # Tính penalty cho prediction ngắn - áp dụng cả training và eval
    #     if self.args.predict_with_generate and model.training:
    #         try:
    #             # Chỉ áp dụng penalty 10% các step để tránh quá chậm
    #             if torch.rand(1).item() < 0.2:
    #                 # Sinh prediction
    #                 generated_ids = model.generate(
    #                     inputs["input_features"],
    #                     attention_mask=inputs.get("attention_mask", None),
    #                     max_length=self.args.generation_max_length,
    #                     num_beams=1,
    #                     forced_decoder_ids=model.generation_config.forced_decoder_ids,
    #                     suppress_tokens=[],  # Không suppress EOS để cho phép dừng
    #                 )
                    
    #                 # Độ dài prediction và label
    #                 pred_lens = (generated_ids != model.config.pad_token_id).sum(dim=1).float()
    #                 label_lens = (inputs["labels"] != -100).sum(dim=1).float()
                    
    #                 # Tính tỷ lệ độ dài
    #                 length_ratio = pred_lens / label_lens.clamp(min=1.0)
                    
    #                 # Phạt nặng nếu prediction quá ngắn (< 0.5 độ dài label)
    #                 short_penalty = torch.where(
    #                     length_ratio < 0.5,
    #                     (0.5 - length_ratio) * 10.0,  # Phạt nặng
    #                     torch.zeros_like(length_ratio)
    #                 ).mean()
                    
    #                 # Phạt nhẹ nếu prediction hơi ngắn (0.5-0.8 độ dài label)
    #                 medium_penalty = torch.where(
    #                     (length_ratio >= 0.5) & (length_ratio < 0.8),
    #                     (0.8 - length_ratio) * 2.0,  # Phạt nhẹ
    #                     torch.zeros_like(length_ratio)
    #                 ).mean()

    #                 long_penalty = torch.where(
    #                     length_ratio >= 1.1,
    #                     (length_ratio-1.1) * 2.0,  # Phạt nhẹ
    #                     torch.zeros_like(length_ratio)
    #                 ).mean()

    #                 very_long_penalty = torch.where(
    #                     length_ratio >= 1.5,
    #                     (length_ratio-1.5) * 10.0,  # Phạt nặng
    #                     torch.zeros_like(length_ratio)
    #                 ).mean()
                    
                    
    #                 # Tổng penalty
    #                 penalty = short_penalty + medium_penalty + long_penalty + very_long_penalty
    #                 loss = loss + penalty
                    
    #         except Exception:
    #             # Nếu generate lỗi, không phạt
    #             pass

    #     if return_outputs:
    #         return loss, outputs
    #     return loss
    # def create_scheduler(self, num_training_steps: int, optimizer=None):
    #     optimizer_to_use = self.optimizer if hasattr(self, 'optimizer') else optimizer
        
    #     from torch.optim.lr_scheduler import ReduceLROnPlateau
        
    #     # Tạo scheduler nhưng chưa set threshold cụ thể
    #     self.lr_scheduler = ReduceLROnPlateau(
    #         optimizer_to_use,
    #         mode="min",
    #         factor=0.5,
    #         patience=1,
    #         threshold=1e-4, 
    #         min_lr=1e-7
    #     )
        
    #     # Lưu lại để điều chỉnh sau
    #     self.scheduler_optimizer = optimizer_to_use
    #     print("🔍 Config scheduler")
    #     return self.lr_scheduler

    # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    #     print("🔍 Custom evaluate method called")
    #     metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    #     print("📊 Metrics returned:", metrics)
        
    #     # ĐIỀU CHỈNH THRESHOLD THEO METRIC
    #     if "eval_cer" in metrics:
    #         # Dùng threshold lớn cho CER
    #         self.lr_scheduler.threshold = 0.005
    #         self.lr_scheduler.threshold_mode = 'abs'
    #         print(f"📉 Scheduler step with CER: {metrics['eval_cer']:.4f}, threshold: 0.005")
    #         self.lr_scheduler.step(metrics["eval_cer"])
    #     else:
    #         # Dùng threshold nhỏ cho Loss
    #         self.lr_scheduler.threshold = 1e-4
    #         self.lr_scheduler.threshold_mode = 'rel'
    #         print(f"📉 Scheduler step with Loss: {metrics['eval_loss']:.4f}, threshold: 1e-4")
    #         self.lr_scheduler.step(metrics["eval_loss"])
        
    #     return metrics
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Đảm bảo tất cả token IDs là integers
        self._ensure_integer_token_ids(model)
        
        # Gọi prediction_step gốc
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
    
    def _ensure_integer_token_ids(self, model):
        """Đảm bảo tất cả token ID là số nguyên để tránh lỗi slice"""
        # Model config
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is not None:
            model.config.pad_token_id = int(model.config.pad_token_id)
        if hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
            model.config.eos_token_id = int(model.config.eos_token_id)
        if hasattr(model.config, 'decoder_start_token_id') and model.config.decoder_start_token_id is not None:
            model.config.decoder_start_token_id = int(model.config.decoder_start_token_id)
        
        # Generation config
        if hasattr(model.generation_config, 'pad_token_id') and model.generation_config.pad_token_id is not None:
            model.generation_config.pad_token_id = int(model.generation_config.pad_token_id)
        eos_token_id = model.generation_config.eos_token_id
        if isinstance(eos_token_id, list) and eos_token_id:
            model.generation_config.eos_token_id = int(eos_token_id[0])
        else:
            model.generation_config.eos_token_id = int(eos_token_id)
        if hasattr(model.generation_config, 'decoder_start_token_id') and model.generation_config.decoder_start_token_id is not None:
            model.generation_config.decoder_start_token_id = int(model.generation_config.decoder_start_token_id)
        
        # Suppress tokens
        if hasattr(model.generation_config, 'suppress_tokens'):
            suppress_tokens = model.generation_config.suppress_tokens
            if suppress_tokens:
                model.generation_config.suppress_tokens = [int(x) for x in suppress_tokens]

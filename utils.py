import re
from typing import Any, Dict
import numpy as np
from tqdm import tqdm
# --- Xử lý Timestamp ---

def convert_global_to_local_timestamps(text: str, offset: float) -> str:
    """
    Chuyển đổi timestamp toàn cục thành timestamp cục bộ (phạm vi 0-30s) và làm tròn theo bội số 0.02s.
    Đầu ra: <|start_local|> text <|end_local|>
    """
    def repl(match):
        global_time = float(match.group(1))
        local_time = global_time - offset
        local_time = max(0.0, min(30.0, local_time))
        local_time = round(local_time / 0.02) * 0.02
        return f"<|{local_time:.2f}|>"

    return re.sub(r"<\|([\d.]+)\|>", repl, text)


# --- Làm sạch và xử lý văn bản ---

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản để tính toán CER.
    """
    text = re.sub(r"<\|[^|]*\|>", "", text)  # Remove timestamps
    text = re.sub(r"\s+", "", text)  # Remove whitespace
    text = re.sub(r"<\|pad\|>", "", text)
    text = re.sub(
        r"[^\w\u4e00-\u9fff]", "", text
    )  # Keep Chinese, letters, numbers
    return text.strip()


# --- Kiểm tra tính hợp lệ ---

def is_valid(example: Dict[str, Any]) -> bool:
    """
    Kiểm tra xem một mẫu âm thanh có lỗi không.
    """
    try:
        arr = example["audio"]["array"]
        return arr is not None and len(arr) > 1000 and isinstance(arr, np.ndarray)
    except Exception:
        return False


# --- Áp dụng timestamp cục bộ ---

def apply_local_timestamps(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Thêm cột `chinese_local` với timestamp cục bộ vào dataset.
    """
    offset = batch.get("start_time", 0)
    batch["chinese_local"] = convert_global_to_local_timestamps(
        batch["chinese_text"], offset
    )
    return batch

# --- Cấu hình Tokenizer ---

def config_tokenizer(tokenizer):
    """
    Thêm token đặc biệt và thiết lập prefix cho tokenizer.
    """
    # Đặt pad token mới
    # tokenizer.pad_token = "<|pad|>"
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    
    # Cấu hình prefix tokens
    tokenizer.set_prefix_tokens(language="zh", task="transcribe", predict_timestamps=True)
    
    # print(f"Tokenizer pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    return tokenizer

def delete_all_checkpoints(output_dir):
    """Xóa tất cả thư mục checkpoint"""
    import shutil
    import os
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if item.startswith("checkpoint") and os.path.isdir(item_path):
            print(f"🗑️ Deleting checkpoint: {item}")
            shutil.rmtree(item_path)
    
    # Kiểm tra lại xem còn checkpoint nào không
    remaining_checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint")]
    if not remaining_checkpoints:
        print("✅ All checkpoints deleted successfully")
    else:
        print(f"⚠️ Still found checkpoints: {remaining_checkpoints}")


def backup_and_delete_all_checkpoints(output_dir, backup_dir):
    """Sao chép tất cả thư mục checkpoint sang backup_dir rồi xóa khỏi output_dir"""
    import shutil
    import os

    # Tạo thư mục backup nếu chưa có
    os.makedirs(backup_dir, exist_ok=True)

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if item.startswith("checkpoint") and os.path.isdir(item_path):
            backup_path = os.path.join(backup_dir, item)

            # Sao chép checkpoint sang thư mục backup
            print(f"📦 Backing up checkpoint: {item} ➡️ {backup_path}")
            shutil.copytree(item_path, backup_path)

            # Xóa checkpoint sau khi sao lưu
            print(f"🗑️ Deleting checkpoint: {item}")
            shutil.rmtree(item_path)

    # Kiểm tra lại xem còn checkpoint nào không
    remaining_checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint")]
    if not remaining_checkpoints:
        print("✅ All checkpoints backed up and deleted successfully")
    else:
        print(f"⚠️ Still found checkpoints: {remaining_checkpoints}")

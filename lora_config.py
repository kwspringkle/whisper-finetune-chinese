from peft import LoraConfig, get_peft_model

# Cấu hình LoRA cho Whisper Medium
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

def wrap_with_lora(model):
    """Apply LoRA to Whisper model"""
    return get_peft_model(model, lora_config)

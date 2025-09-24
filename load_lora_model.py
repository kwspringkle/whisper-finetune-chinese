import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

def verify_model_changes(finetuned_model, base_model_name="openai/whisper-medium", sample_layers=5):
    """
    Verify rằng model đã được fine-tune bằng cách so sánh weights với base model
    """
    print("\n🔍 Verifying model changes against base model...")
    
    # Load base model để so sánh
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    
    # Lấy một vài layers quan trọng để check
    important_layers = [
        "model.encoder.layers.0.self_attn.q_proj.weight",
        "model.encoder.layers.5.self_attn.v_proj.weight", 
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.5.cross_attn.q_proj.weight",
        "proj_out.weight"
    ]
    
    changes_found = 0
    total_checked = 0
    
    print("📊 Checking weights in key layers:")
    
    for layer_name in important_layers:
        try:
            # Get weights từ cả 2 models
            finetuned_weight = dict(finetuned_model.named_parameters())[layer_name]
            base_weight = dict(base_model.named_parameters())[layer_name]
            
            # Tính difference
            diff = torch.abs(finetuned_weight - base_weight)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            # Check nếu có thay đổi đáng kể
            changed = max_diff > 1e-6
            if changed:
                changes_found += 1
                status = "✅ CHANGED"
            else:
                status = "❌ SAME"
            
            total_checked += 1
            print(f"├─ {layer_name.split('.')[-2:]}: {status}")
            print(f"│  ├─ Max diff: {max_diff:.2e}")
            print(f"│  └─ Mean diff: {mean_diff:.2e}")
            
        except KeyError:
            print(f"├─ {layer_name}: ⚠️ Not found")
        except Exception as e:
            print(f"├─ {layer_name}: ❌ Error - {e}")
    
    # Summary
    print("\n📈 Verification Summary:")
    print(f"├─ Layers checked: {total_checked}")
    print(f"├─ Layers changed: {changes_found}")
    print(f"└─ Change ratio: {(changes_found/total_checked)*100 if total_checked > 0 else 0:.1f}%")
    
    if changes_found > 0:
        print("✅ Model HAS been fine-tuned! (Weights different from base)")
    else:
        print("⚠️  Model appears SAME as base (No significant weight changes)")
    
    # Cleanup
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return changes_found > 0

def load_finetuned_model(model_path="./whisper-chinese-timestamp", verify_changes=True):
    """
    Load Whisper model đã fine-tune (merged LoRA)
    
    Args:
        model_path: Path hoặc Hub model name
        verify_changes: Có verify model đã thay đổi so với base không
    
    Returns:
        model, processor
    """
    print(f"🔄 Loading fine-tuned Whisper model: {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    print("✅ Model loaded successfully!")
    
    # Verify changes nếu cần
    if verify_changes:
        is_changed = verify_model_changes(model)
        if not is_changed:
            print("⚠️  Warning: Model may not have been properly fine-tuned!")
    
    return model, processor

def load_lora_adapter_only(base_model_name="openai/whisper-medium", lora_adapter_path="./whisper-chinese-timestamp_lora_adapter"):
    """
    Load Whisper model với LoRA adapter riêng (backup)
    
    Args:
        base_model_name: Tên base model (openai/whisper-medium)
        lora_adapter_path: Path đến LoRA adapter backup
    
    Returns:
        model, processor
    """
    from peft import PeftModel
    
    print(f"🔄 Loading base model: {base_model_name}")
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    
    print(f"🔄 Loading LoRA adapter: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Load processor from main model directory
    processor_path = lora_adapter_path.replace("_lora_adapter", "")
    print(f"🔄 Loading processor from: {processor_path}")
    processor = WhisperProcessor.from_pretrained(processor_path)
    
    # Merge LoRA weights để inference nhanh hơn (optional)
    print("🔄 Merging LoRA weights...")
    model = model.merge_and_unload()
    
    print("✅ Model loaded successfully!")
    return model, processor

def create_pipeline(model_path="./whisper-chinese-timestamp"):
    """
    Tạo pipeline ASR đơn giản
    """
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        return_timestamps=True,
        chunk_length_s=30,
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe

def transcribe_audio(model, processor, audio_path):
    """
    Transcribe audio file sử dụng model LoRA
    """
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    
    # Generate
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="chinese",
            task="transcribe",
            return_timestamps=True
        )
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    # Example usage - load merged model (recommended)
    model, processor = load_finetuned_model()
    
    # Or create pipeline (easiest)
    pipe = create_pipeline()
    
    # Test transcription
    # result = transcribe_audio(model, processor, "test_audio.wav")
    # print(f"Transcription: {result}")
    
    # Or use pipeline
    # result = pipe("test_audio.wav")
    # print(f"Pipeline result: {result}")
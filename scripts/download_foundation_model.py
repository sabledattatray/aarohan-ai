"""Download and verify foundation model weights for local offline AAROHAN operation."""

import os
import sys

import torch

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[*] Initializing download for: {model_name}")

    from huggingface_hub import snapshot_download
    local_path = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
        resume_download=True,
    )
    print(f"[+] Download complete: {local_path}")

    print("[*] Testing model instantiation with transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[*] Loading model on {device} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=dtype).to(device)
    model.eval()
    print("[+] Model loaded successfully!")

    messages = [
        {"role": "system", "content": "You are AAROHAN, an independent AI assistant."},
        {"role": "user", "content": "Can you help me in data analysis?"}
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_str], return_tensors="pt").to(device)

    print("[*] Generating test response...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.7, do_sample=True)
    resp = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("=" * 60)
    print("GENERATED RESPONSE:")
    print(resp)
    print("=" * 60)

if __name__ == "__main__":
    main()

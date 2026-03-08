#!/usr/bin/env python3
"""
Test chat prompt formatting for a model (e.g. Mistral).
Run: python -m graviton_ui.scripts.test_chat_prompt mistralai/Mistral-7B-Instruct-v0.3
Shows the exact prompt string that would be sent to the model.
"""
import sys

def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    print(f"Loading tokenizer for {model_id}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    except Exception:
        local_dir = snapshot_download(model_id, allow_patterns=["*.json", "*.model", "tokenizer*"])
        tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=False)

    has_template = bool(getattr(tok, "chat_template", None))
    print(f"chat_template present: {has_template}\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello how are you"},
    ]
    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("Prompt that will be sent to the model:")
        print("-" * 60)
        print(prompt)
        print("-" * 60)
        print(f"Length: {len(prompt)} chars")
    except Exception as e:
        print(f"apply_chat_template failed: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

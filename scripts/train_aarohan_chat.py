"""CLI executable to train or fine-tune Aarohan AI on consumer hardware.

Usage:
    python scripts/train_aarohan_chat.py --epochs 3 --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aarohan.training.fine_tune_lora import AarohanLoRATrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_aarohan")


def main():
    parser = argparse.ArgumentParser(description="Train or align Aarohan AI with LoRA on local GPU")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset", type=str, default="data/training/chat_alignment_dataset.json", help="Path to JSON dataset")
    parser.add_argument("--output", type=str, default="checkpoints/aarohan-chat-lora", help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank r")

    args = parser.parse_args()

    config = TrainingConfig(
        base_model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.rank,
    )

    trainer = AarohanLoRATrainer(config)
    print("=" * 60)
    print("✦ AAROHAN AI - ChatGPT Alignment & LoRA Fine-Tuning")
    print(f"Base Model:  {config.base_model_name}")
    print(f"Dataset:     {config.dataset_path}")
    print(f"Output Dir:  {config.output_dir}")
    print(f"Device:      {config.device}")
    print("=" * 60)

    meta = trainer.train()
    print("\n✅ Training Complete!")
    print(f"Duration:   {meta['duration_seconds']} seconds")
    print(f"Final Loss: {meta['final_loss']:.4f}")
    print(f"Adapter:    {config.output_dir}")


if __name__ == "__main__":
    main()

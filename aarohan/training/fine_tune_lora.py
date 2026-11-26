"""AAROHAN AI - LoRA Conversational Fine-Tuning & Alignment Engine.

Engineered specifically for NVIDIA GeForce RTX 2060 (6GB VRAM) and CPU fallback.
Fine-tunes causal language models (Qwen2.5, Llama, TinyLlama) into ChatGPT-level
conversational assistants using parameter-efficient Low-Rank Adaptation (LoRA).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("data/training/chat_alignment_dataset.json")
DEFAULT_OUTPUT_DIR = Path("checkpoints/aarohan-chat-lora")


@dataclass
class TrainingConfig:
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = str(DEFAULT_DATASET_PATH)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ChatDataset(Dataset):
    """Tokenized conversational dataset for multi-turn instruction tuning."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer: Any, max_length: int = 512):
        self.examples = []
        for item in data:
            messages = item.get("messages", [])
            if not messages:
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask padding in loss computation
            labels[attention_mask == 0] = -100
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class AarohanLoRATrainer:
    """Manages parameter-efficient fine-tuning on consumer hardware."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()

    def load_dataset(self) -> List[Dict[str, Any]]:
        path = Path(self.config.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Training dataset not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def train(self) -> Dict[str, Any]:
        """Execute LoRA alignment on GPU/CPU."""
        t_start = time.time()
        logger.info("Initializing LoRA training for %s on %s", self.config.base_model_name, self.config.device)

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Load Dataset
        raw_data = self.load_dataset()
        dataset = ChatDataset(raw_data, tokenizer, max_length=self.config.max_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 3. Load Base Model in FP16 on GPU (low memory footprint < 1.5 GB VRAM)
        dtype = torch.float16 if self.config.device == "cuda" else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.config.device)

        # 4. Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        model = get_peft_model(base_model, peft_config)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            "Trainable params: %d / %d (%.2f%%)",
            trainable_params,
            all_params,
            100 * trainable_params / all_params,
        )

        # 5. Training Loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        model.train()

        loss_history = []
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            step = 0
            optimizer.zero_grad()

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                step += 1
                if step % self.config.gradient_accumulation_steps == 0 or step == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)
            logger.info("Epoch %d/%d - Loss: %.4f", epoch, self.config.epochs, avg_loss)

        # 6. Save Adapter Weights
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        # Metadata record
        training_meta = {
            "base_model": self.config.base_model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "epochs": self.config.epochs,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "duration_seconds": round(time.time() - t_start, 2),
            "trainable_params": trainable_params,
            "total_params": all_params,
            "device": self.config.device,
        }
        with open(out_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2)

        logger.info("LoRA adapter saved successfully to %s in %.2fs", out_dir, training_meta["duration_seconds"])
        return training_meta

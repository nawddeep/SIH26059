#!/usr/bin/env python3
"""
scripts/train_local_model.py

Hardware-aware Apple Silicon LoRA fine-tuning pipeline for IMPALA Local Dashboard.
Trains a local Qwen model on synthetic polar maritime operational dialogues.

Features:
- MLX-LM LoRA adapter training utilizing Apple Silicon Metal unified memory.
- Gradient checkpointing & accumulation for memory efficiency.
- Auto-validation & periodic checkpointing.
- Model adapter output saved to models/impala-qwen3-8b/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "synthetic" / "training"
MODELS_DIR = BASE_DIR / "models" / "impala-qwen3-8b"


def parse_args():
    parser = argparse.ArgumentParser(description="Train local Qwen LoRA adapter for IMPALA Local Dashboard.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-4bit", help="Base model identifier or local path")
    parser.add_argument("--data", type=str, default=str(DATA_DIR), help="Path to training data directory containing train.jsonl, valid.jsonl")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR), help="Output directory for trained LoRA adapter")
    parser.add_argument("--iters", type=int, default=300, help="Total training iterations")
    parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size")
    parser.add_argument("--grad-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers to fine-tune with LoRA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("  IMPALA LOCAL MODEL LoRA TRAINING PIPELINE")
    print("==================================================")
    print(f"Base Model:             {args.model}")
    print(f"Data Directory:         {args.data}")
    print(f"Adapter Output:         {args.output_dir}")
    print(f"Iterations:             {args.iters}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Gradient Accumulation:  {args.grad_accumulation}")
    print(f"Learning Rate:          {args.learning_rate}")
    print(f"Max Sequence Length:    {args.max_seq_length}")
    print(f"LoRA Target Layers:     {args.num_layers}")
    print(f"Seed:                   {args.seed}")
    print("==================================================")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", args.data,
        "--adapter-path", str(out_dir),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--grad-accumulation-steps", str(args.grad_accumulation),
        "--learning-rate", str(args.learning_rate),
        "--max-seq-length", str(args.max_seq_length),
        "--num-layers", str(args.num_layers),
        "--grad-checkpoint",
        "--steps-per-report", "20",
        "--steps-per-eval", "50",
        "--val-batches", "10",
        "--save-every", "100",
        "--seed", str(args.seed),
    ]

    print(f"Executing training command:\n{' '.join(cmd)}\n")
    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[FAIL] Training process exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Verify adapter artifact output
    adapter_weights = out_dir / "adapters.safetensors"
    adapter_config = out_dir / "adapter_config.json"

    if not adapter_weights.exists():
        print(f"[FAIL] Expected adapter file missing: {adapter_weights}")
        sys.exit(1)

    print("==================================================")
    print(f"  TRAINING COMPLETED SUCCESSFULLY in {elapsed:.1f}s")
    print("==================================================")
    print(f"Adapter artifact generated at: {adapter_weights} ({adapter_weights.stat().st_size / 1024 / 1024:.2f} MB)")
    if adapter_config.exists():
        print(f"Adapter config: {adapter_config}")


if __name__ == "__main__":
    main()

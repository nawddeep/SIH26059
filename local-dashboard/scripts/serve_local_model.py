#!/usr/bin/env python3
"""
scripts/serve_local_model.py

Starts the local OpenAI-compatible inference server using the fine-tuned
IMPALA Qwen model and trained LoRA adapter.

Endpoint: http://127.0.0.1:8080/v1/chat/completions
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "impala-qwen3-8b"


def main():
    parser = argparse.ArgumentParser(description="Serve IMPALA Local Fine-Tuned Model")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-4bit", help="Base model")
    parser.add_argument("--adapter-path", type=str, default=str(MODELS_DIR), help="Trained adapter path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (localhost only)")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    args = parser.parse_args()

    print("==================================================")
    print("  IMPALA LOCAL MODEL SERVER STARTUP")
    print("==================================================")
    print(f"Base Model:    {args.model}")
    print(f"Adapter Path:  {args.adapter_path}")
    print(f"Endpoint:      http://{args.host}:{args.port}/v1/chat/completions")
    print("==================================================")

    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", args.model,
        "--adapter-path", args.adapter_path,
        "--host", args.host,
        "--port", str(args.port),
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()

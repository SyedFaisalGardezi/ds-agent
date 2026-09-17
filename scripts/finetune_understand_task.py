"""
LoRA fine-tuning for the understand_task (problem scoping) step.

Two training targets:
  --target scope   → fine-tune qwen3.6:latest on structured SCOPE ANALYSIS format
                     Dataset: data/finetuning/understand_task_scope.jsonl
  --target extract → fine-tune on flat JSON RawSpec extraction format
                     Dataset: data/finetuning/understand_task_extract.jsonl (optional)

Usage:
    # Scope model (recommended — teaches structured reasoning output):
    python scripts/finetune_understand_task.py \\
        --data data/finetuning/understand_task_scope.jsonl \\
        --output outputs/scope_lora \\
        --base_model Qwen/Qwen2.5-7B-Instruct \\
        --target scope

    # Or let the script pick the dataset automatically:
    python scripts/finetune_understand_task.py --target scope

After training, convert to GGUF for Ollama:
    python -m mlx_lm.fuse --model outputs/scope_lora/merged
    python llama.cpp/convert_hf_to_gguf.py outputs/merged \\
        --outfile scope_model.gguf --outtype q4_k_m
    ollama create understand-scope -f Modelfile
    # Then set env var: DSAGENT_SCOPE_MODEL=understand-scope
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> str:
    """Return backend: 'transformers' | 'mlx'."""
    # Apple Silicon: prefer mlx-lm (much faster, no CUDA needed)
    if platform.processor() == "arm" and platform.system() == "Darwin":
        try:
            import mlx  # noqa: F401
            import mlx_lm  # noqa: F401
            log.info("Apple Silicon detected — using mlx-lm backend")
            return "mlx"
        except ImportError:
            log.warning(
                "mlx / mlx-lm not found. Falling back to transformers+PEFT. "
                "Install with: pip install mlx-lm"
            )

    # Otherwise require transformers + peft + trl
    missing = []
    for pkg in ("transformers", "peft", "trl", "torch", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s — install with pip install %s", missing, " ".join(missing))
        sys.exit(1)

    return "transformers"


# ── dataset loading ───────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d training examples from %s", len(records), path)
    return records


def _validate_records(records: list[dict]) -> list[dict]:
    valid = []
    for i, rec in enumerate(records):
        msgs = rec.get("messages", [])
        roles = [m.get("role") for m in msgs]
        if roles == ["system", "user", "assistant"]:
            valid.append(rec)
        else:
            log.warning("Record %d has unexpected roles %s — skipped", i, roles)
    if not valid:
        log.error("No valid records found. Check JSONL format.")
        sys.exit(1)
    log.info("%d / %d records are valid chat-format examples", len(valid), len(records))
    return valid


# ── MLX-LM backend (Apple Silicon) ───────────────────────────────────────────

def _resolve_mlx_model_path(base_model: str, output_dir: Path) -> str:
    """Resolve the model path for mlx-lm.

    Handles three input forms:
      1. HuggingFace repo ID — pass through unchanged (mlx-lm downloads it).
      2. Local directory with safetensors/config.json — pass through.
      3. Local GGUF file — convert to MLX format first (requires mlx-lm ≥0.20).
      4. Ollama model name (e.g. "model:tag") — locate the GGUF blob in
         ~/.ollama/models/ and treat as case 3.
    """
    import subprocess  # noqa: PLC0415

    p = Path(base_model)

    # Case 2: existing local MLX directory
    if p.is_dir() and (p / "config.json").exists():
        log.info("Using local MLX model directory: %s", p)
        return str(p)

    # Case 3: existing local GGUF file → convert
    if p.is_file() and p.suffix in (".gguf", ".bin"):
        return _convert_gguf_to_mlx(p, output_dir)

    # Case 4: Ollama model name → find GGUF blob
    if ":" in base_model and not base_model.startswith("http") and not Path(base_model).exists():
        gguf_path = _find_ollama_gguf(base_model)
        if gguf_path:
            return _convert_gguf_to_mlx(gguf_path, output_dir)
        log.warning(
            "Could not locate Ollama GGUF for '%s'. "
            "Passing the name directly to mlx-lm (may fail). "
            "Pull the model first: ollama pull %s",
            base_model, base_model,
        )

    # Case 1 (default): HuggingFace repo ID — pass through
    return base_model


def _find_ollama_gguf(model_name: str) -> Path | None:
    """Locate the GGUF blob file for an Ollama model."""
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            ["ollama", "show", "--modelfile", model_name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log.warning("ollama show failed: %s", result.stderr.strip())
            return None
        for line in result.stdout.splitlines():
            # Modelfile line: FROM /path/to/blob or FROM sha256:...
            if line.strip().startswith("FROM "):
                src = line.strip()[5:].strip()
                p = Path(src)
                if p.exists():
                    log.info("Found Ollama GGUF at %s", p)
                    return p
                # Try ~/.ollama/models/blobs/<sha>
                blob_dir = Path.home() / ".ollama" / "models" / "blobs"
                sha = src.replace("sha256:", "sha256-")
                candidate = blob_dir / sha
                if candidate.exists():
                    log.info("Found Ollama blob at %s", candidate)
                    return candidate
    except Exception as e:
        log.warning("Could not locate Ollama GGUF: %s", e)
    return None


def _convert_gguf_to_mlx(gguf_path: Path, output_dir: Path) -> str:
    """Convert a GGUF model to MLX format using mlx_lm.convert."""
    import subprocess  # noqa: PLC0415

    mlx_model_dir = output_dir / "mlx_base_model"
    if mlx_model_dir.exists() and (mlx_model_dir / "config.json").exists():
        log.info("MLX base model already exists at %s — skipping conversion", mlx_model_dir)
        return str(mlx_model_dir)

    log.info("Converting GGUF → MLX format: %s → %s", gguf_path, mlx_model_dir)
    mlx_model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", str(gguf_path),
        "--mlx-path", str(mlx_model_dir),
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error(
            "GGUF conversion failed (exit %d). "
            "Try passing a HuggingFace repo ID with --base_model instead.",
            result.returncode,
        )
        sys.exit(result.returncode)
    log.info("Conversion complete → %s", mlx_model_dir)
    return str(mlx_model_dir)


def _train_mlx(
    records: list[dict],
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    lora_r: int,
    max_seq_len: int,
) -> None:
    """Fine-tune using mlx-lm LoRA on Apple Silicon.

    Supports:
      - HuggingFace model IDs (downloaded automatically)
      - Local MLX model directories
      - Local GGUF files (converted to MLX format first)
      - Ollama model names (GGUF located from ~/.ollama/models/blobs/ and converted)

    NOTE: Q4_K_M GGUFs cannot be fine-tuned directly. mlx-lm will load the GGUF
    but keep the base weights frozen and quantized; only the LoRA adapter layers
    are trained in FP16. This is equivalent to QLoRA and is memory-efficient on
    32 GB M1 Pro.
    """
    try:
        from mlx_lm import load as mlx_load  # noqa: F401 — verify import
    except ImportError as e:
        log.error("mlx-lm not installed: %s\nInstall with: pip install mlx-lm", e)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model to an mlx-compatible path
    resolved_model = _resolve_mlx_model_path(base_model, output_dir)

    # Convert chat-format JSONL → mlx-lm prompt/completion format
    mlx_data_path = output_dir / "mlx_train.jsonl"
    _write_mlx_train_data(records, resolved_model, mlx_data_path)

    # Qwen3-30B-A3B has 94 transformer layers; LoRA on top 16 is a good balance.
    # Increase lora_layers to 32 if you have RAM headroom.
    lora_layers = 16

    lora_config = {
        "model": resolved_model,
        "train": True,
        "data": str(mlx_data_path.parent),
        "seed": 42,
        "lora_layers": lora_layers,
        "batch_size": batch_size,
        "iters": max(80, len(records) * epochs),
        "val_batches": 5,
        "learning_rate": lr,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "save_every": 100,
        "adapter_path": str(output_dir / "adapters"),
        "max_seq_length": max_seq_len,
        "lora_parameters": {
            "rank": lora_r,
            "alpha": lora_r * 2,
            "dropout": 0.05,
            "scale": 10.0,
        },
    }

    config_path = output_dir / "lora_config.json"
    config_path.write_text(json.dumps(lora_config, indent=2))
    log.info("Saved mlx-lm lora config to %s", config_path)

    log.info("Starting mlx-lm LoRA training (%d iters, lora_layers=%d)…",
             lora_config["iters"], lora_layers)
    import subprocess  # noqa: PLC0415
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", resolved_model,
        "--train",
        "--data", str(mlx_data_path.parent),
        "--batch-size", str(batch_size),
        "--iters", str(lora_config["iters"]),
        "--learning-rate", str(lr),
        "--lora-layers", str(lora_layers),
        "--adapter-path", lora_config["adapter_path"],
        "--max-seq-length", str(max_seq_len),
        "--seed", "42",
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("mlx-lm training failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    log.info("Training complete. Adapters saved to %s", lora_config["adapter_path"])
    _print_mlx_next_steps(output_dir, resolved_model)


def _write_mlx_train_data(records: list[dict], model_path: str, out_path: Path) -> None:
    """Convert chat-format records to mlx-lm prompt/completion JSONL."""
    # Try to load tokenizer for proper chat template application.
    # Fall back to a simple join if the tokenizer can't be loaded.
    tokenizer = None
    try:
        from mlx_lm import load as mlx_load  # noqa: PLC0415
        _, tokenizer = mlx_load(model_path)
    except Exception:
        pass

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            pass

    log.info("Writing %d training examples to %s …", len(records), out_path)
    with out_path.open("w") as f:
        for rec in records:
            msgs = rec["messages"]
            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            assistant_content = next(
                m["content"] for m in msgs if m["role"] == "assistant"
            )
            if tokenizer is not None:
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_msgs, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_text = "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in prompt_msgs
                    )
            else:
                prompt_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in prompt_msgs
                )
            f.write(json.dumps({"prompt": prompt_text, "completion": assistant_content}) + "\n")
    log.info("Wrote mlx-lm training data to %s", out_path)


def _print_mlx_next_steps(output_dir: Path, base_model: str) -> None:
    adapter_path = output_dir / "adapters"
    merged_path = output_dir / "merged"
    ollama_model_name = "understand-scope"
    gguf_path = output_dir / "understand_scope.gguf"
    print("\n" + "=" * 60)
    print("MLX-LM TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nAdapters saved to: {adapter_path}")
    print("\nNext steps — merge, quantize, and deploy to Ollama:")
    print(f"""
  # 1. Fuse LoRA adapter into the base model weights
  python -m mlx_lm.fuse \\
      --model {base_model} \\
      --adapter-path {adapter_path} \\
      --save-path {merged_path}

  # 2. Convert merged model to GGUF Q4_K_M (requires llama.cpp)
  python llama.cpp/convert_hf_to_gguf.py {merged_path} \\
      --outfile {gguf_path} \\
      --outtype q4_k_m

  # 3. Create Ollama Modelfile
  cat > {output_dir}/Modelfile <<'EOF'
FROM {gguf_path}
PARAMETER temperature 0.1
PARAMETER top_p 0.9
SYSTEM "You are an expert data science problem analyst. Reason carefully and output a structured SCOPE ANALYSIS block."
EOF

  # 4. Register model in Ollama
  ollama create {ollama_model_name} -f {output_dir}/Modelfile
  ollama run {ollama_model_name}

  # 5. Point ds-agent at the fine-tuned model
  export DSAGENT_SCOPE_MODEL={ollama_model_name}
  # Or set in your shell profile / launchd plist for persistence.
""")


# ── Transformers + PEFT + TRL backend ────────────────────────────────────────

def _train_transformers(
    records: list[dict],
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    max_seq_len: int,
) -> None:
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer  # noqa: PLC0415

    # Detect device
    if torch.cuda.is_available():
        device_map = "auto"
        log.info("Using CUDA")
    elif torch.backends.mps.is_available():
        # MPS works for inference but PEFT training has issues; warn
        device_map = "cpu"
        log.warning(
            "MPS detected but PEFT training has known issues on MPS. "
            "Training on CPU (slow). For Apple Silicon prefer mlx-lm backend."
        )
    else:
        device_map = "cpu"
        log.info("No GPU detected — training on CPU (very slow for 7B model)")

    # Quantisation config (4-bit) — only available with CUDA
    bnb_config = None
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401, PLC0415
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            log.info("4-bit quantisation enabled via bitsandbytes")
        except ImportError:
            log.warning("bitsandbytes not installed — loading model in fp16")

    log.info("Loading base model: %s …", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    # LoRA config — target attention + FFN projections in Qwen2.5
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",         # FFN (SwiGLU)
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build HuggingFace Dataset from records using chat template
    def _format(rec: dict) -> str:
        return tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_special_tokens=True
        )

    formatted = [_format(r) for r in records]
    dataset = Dataset.from_dict({"text": formatted})

    # Split off 10% for validation
    split = dataset.train_test_split(test_size=max(1, int(len(dataset) * 0.1)), seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    log.info("Train: %d  Eval: %d", len(train_ds), len(eval_ds))

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=False,
    )

    log.info("Starting SFT training …")
    trainer.train()

    # Save LoRA adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info("LoRA adapter saved to %s", adapter_path)

    _print_transformers_next_steps(output_dir, base_model, adapter_path)


def _print_transformers_next_steps(
    output_dir: Path, base_model: str, adapter_path: Path
) -> None:
    merged_path = output_dir / "merged"
    print("\n" + "=" * 60)
    print("PEFT / SFT TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nLoRA adapter: {adapter_path}")
    print("\nNext steps — merge and export to Ollama:")
    print(f"""
  # 1. Merge adapter into full model weights
  python - <<'PY'
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model = AutoPeftModelForCausalLM.from_pretrained("{adapter_path}", device_map="cpu")
merged = model.merge_and_unload()
merged.save_pretrained("{merged_path}")
AutoTokenizer.from_pretrained("{base_model}").save_pretrained("{merged_path}")
PY

  # 2. Convert to GGUF (requires llama.cpp clone + build)
  python llama.cpp/convert_hf_to_gguf.py {merged_path} \\
      --outfile {output_dir}/understand_task.gguf \\
      --outtype q4_k_m

  # 3. Create Modelfile and load into Ollama
  echo 'FROM {output_dir}/understand_task.gguf' > {output_dir}/Modelfile
  echo 'PARAMETER temperature 0.1' >> {output_dir}/Modelfile
  ollama create understand-task -f {output_dir}/Modelfile
  ollama run understand-task
""")


# ── quick inference test ──────────────────────────────────────────────────────

def _quick_test(base_model: str, records: list[dict], backend: str) -> None:
    """Run a single inference pass with the base model to confirm the setup works."""
    log.info("Running quick inference test on base model …")
    sample = records[0]["messages"]
    prompt_msgs = [m for m in sample if m["role"] != "assistant"]

    if backend == "mlx":
        try:
            from mlx_lm import load, generate  # noqa: PLC0415
            model, tokenizer = load(base_model)
            prompt = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
            log.info("Sample model output (first 300 chars):\n%s", response[:300])
        except Exception as e:
            log.warning("Quick test failed (non-fatal): %s", e)
    else:
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa: PLC0415
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True,
            )
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
            prompt = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            out = pipe(prompt)[0]["generated_text"]
            log.info("Sample model output (first 300 chars):\n%s", out[-300:])
        except Exception as e:
            log.warning("Quick test failed (non-fatal): %s", e)


# ── CLI ───────────────────────────────────────────────────────────────────────

_TARGET_DEFAULTS = {
    "scope": (
        Path("data/finetuning/understand_task_scope.jsonl"),
        Path("outputs/scope_lora"),
    ),
    "extract": (
        Path("data/finetuning/understand_task_extract.jsonl"),
        Path("outputs/extract_lora"),
    ),
}


# Base model for fine-tuning.
# The Ollama model "Qwen3.6-35B-A3B-Opus4.7-Reasoning-Distilled:q4km" is the
# inference target.  For fine-tuning we need the model in one of:
#   (a) HuggingFace format    — pass a HF repo ID, e.g. "Qwen/Qwen3-30B-A3B"
#   (b) Local MLX directory   — if already converted with mlx_lm.convert
#   (c) Local GGUF file path  — mlx-lm ≥0.20 can load GGUFs directly
#
# If you have the GGUF on disk, set:
#   FINETUNE_BASE_MODEL=/path/to/Qwen3.6-35B-A3B-Opus4.7-Reasoning-Distilled.gguf
# or pass it via --base_model.
_DEFAULT_BASE_MODEL = (
    os.environ.get("FINETUNE_BASE_MODEL")
    or "Qwen3.6-35B-A3B-Opus4.7-Reasoning-Distilled:q4km"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen3.6-35B-A3B-Opus4.7-Reasoning-Distilled for understand_task"
    )
    p.add_argument(
        "--target",
        choices=["scope", "extract"],
        default="scope",
        help=(
            "scope   → train on structured SCOPE ANALYSIS format (for qwen3.6:latest)\n"
            "extract → train on JSON RawSpec extraction format"
        ),
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the JSONL training file (defaults based on --target)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to save adapter + logs (defaults based on --target)",
    )
    p.add_argument(
        "--base_model",
        default=_DEFAULT_BASE_MODEL,
        help=(
            "Model to fine-tune. Accepts:\n"
            "  - HuggingFace repo ID (e.g. Qwen/Qwen3-30B-A3B)\n"
            "  - Local MLX model directory\n"
            "  - Local GGUF file path (mlx-lm ≥0.20)\n"
            "  Defaults to FINETUNE_BASE_MODEL env var or the Ollama model name."
        ),
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (2x rank recommended)")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument(
        "--test_only",
        action="store_true",
        help="Run a quick inference test without training",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "mlx", "transformers"],
        default="auto",
        help="Training backend (auto = detect Apple Silicon)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Fill data/output defaults from target if not explicitly set
    default_data, default_output = _TARGET_DEFAULTS[args.target]
    if args.data is None:
        args.data = default_data
    if args.output is None:
        args.output = default_output

    if not args.data.exists():
        log.error(
            "Training data not found at %s\n"
            "Generate it first with:\n"
            "  python scripts/generate_finetuning_data.py --mode %s --out %s",
            args.data, args.target, args.data,
        )
        sys.exit(1)

    records = _load_jsonl(args.data)
    records = _validate_records(records)

    if args.backend == "auto":
        backend = _check_deps()
    else:
        backend = args.backend
        _check_deps()  # still validate deps exist

    if args.test_only:
        _quick_test(args.base_model, records, backend)
        return

    log.info(
        "Training config: model=%s  backend=%s  epochs=%d  "
        "batch=%d  lr=%g  lora_r=%d  lora_alpha=%d  max_seq=%d",
        args.base_model, backend, args.epochs, args.batch_size,
        args.lr, args.lora_r, args.lora_alpha, args.max_seq_len,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # Dump run config for reproducibility
    config = {**vars(args), "data": str(args.data), "output": str(args.output)}
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2))

    if backend == "mlx":
        _train_mlx(
            records=records,
            base_model=args.base_model,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_r,
            max_seq_len=args.max_seq_len,
        )
    else:
        _train_transformers(
            records=records,
            base_model=args.base_model,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_seq_len=args.max_seq_len,
        )


if __name__ == "__main__":
    main()

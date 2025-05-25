"""Minimal yet extensible script to benchmark different LoRA BA initialisations
on an RTX-class single-GPU box.

Supported variants so far
------------------------
* B_uniform  : B ~ U(-0.01, 0.01),  A = 0  (paper 2406.08447 vanilla)
* A_uniform  : A ~ U(-0.01, 0.01),  B = 0  (paper hypothesis)

The rest of the pipeline (model, dataset, trainer) is identical, so any
performance gap is attributable to the initialisation scheme.

Tested on:
* CUDA 12.3 + RTX 2070 8 GB

Usage examples
--------------
```bash
# Variant 1: B uniform, A zero (default)
python ba_init_lora.py \
       --init_variant B_uniform \
       --run_name b_uni_a_zero

# Variant 2: A uniform, B zero
python ba_init_lora.py \
       --init_variant A_uniform \
       --run_name a_uni_b_zero

# Swap to a different base LLM (e.g. Mistral-7B)
python ba_init_lora.py \
       --base_model mistralai/Mistral-7B-v0.1 \
       --target_modules q_proj v_proj \
       --init_variant B_uniform
```

After training, TensorBoard logs (loss, learning-rate, eval loss) live under
`runs/<run_name>`.
"""


import argparse
import os
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
import wandb
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer


# --------------------------------------------------------------------------------------
# Configuration dataclass ----------------------------------------------------------------
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    base_model: str = "meta-llama/Meta-Llama-3-8B"  # can be swapped via CLI
    dataset_name: str = "glue"
    dataset_config: str = "sst2"  # GLUE subset
    train_samples: int = 20000
    val_samples: int = 2000
    max_seq_len: int = 256
    rank: int = 8
    alpha: int = 16
    batch_size: int = 4
    grad_accum: int = 4
    num_epochs: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.0
    bf16: bool = True
    init_variant: str = "B_uniform"  # or "A_uniform"
    target_modules: List[str] | None = None  # detect automatically if None
    run_name: str = "lora_ba_init"


# --------------------------------------------------------------------------------------
# Utility: determine device + auto target modules -----------------------------------------
# --------------------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def infer_target_modules(model) -> List[str]:
    """Return sensible LoRA target modules for common decoder-only models."""
    # Llama / Mistral-style: each attn layer has q_proj, k_proj, v_proj, o_proj
    sample_names = [n for n, _ in model.named_modules()][:200]  # cheap lookahead
    if any("q_proj" in n for n in sample_names):
        return ["q_proj", "v_proj"]
    # Falcon: query_key_value + dense
    if any("query_key_value" in n for n in sample_names):
        return ["query_key_value"]
    # Gemma / Gemma-like: same as Llama
    if any("transformer.h." in n for n in sample_names):
        return ["q_proj", "v_proj"]
    # Fallback: blanket LoRA all linear layers (slower!)
    return [".*projection.*", ".*lin.*"]


# --------------------------------------------------------------------------------------
# Custom initialisation -------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def init_lora_weights(model, variant: str, a_low: float = -0.01, a_high: float = 0.01):
    for module in model.modules():
        if isinstance(module, LoraLayer):
            if variant == "B_uniform":
                torch.nn.init.zeros_(module.lora_A["default"].weight)
                torch.nn.init.uniform_(module.lora_B["default"].weight, a=a_low, b=a_high)
            elif variant == "A_uniform":
                torch.nn.init.uniform_(module.lora_A["default"].weight, a=a_low, b=a_high)
                torch.nn.init.zeros_(module.lora_B["default"].weight)
            else:
                raise ValueError(f"Unknown init_variant: {variant}")


# --------------------------------------------------------------------------------------
# Dataset preparation ---------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def build_prompt(example):
    """Turn an SST-2 sentence into an instruction-tuning prompt."""
    sentence = example["sentence"]
    label = example["label"]
    sentiment = "positive" if label == 1 else "negative"
    prompt = f"Classify the sentiment of the following sentence.\n\nSentence: \"{sentence}\"\nSentiment: {sentiment}\n"
    return prompt


# --------------------------------------------------------------------------------------
# Main training pipeline ------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def safe_slice(ds, k):
    """Return first k rows or the whole dataset if k exceeds length."""
    return ds.select(range(min(k, len(ds))))

def run(cfg: Config):
    device = get_device()
    device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if device == "cuda" else device

    print(f"Using device: {device}")

    # ---- Quantised model load (4-bit QLoRA) ----
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     cfg.base_model,
    #     quantization_config=bnb_config,
    #     device_map={"": device},
    # )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- LoRA attach ----
    target_modules = cfg.target_modules or infer_target_modules(model)
    print("Target modules:", target_modules)

    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    def convert_to_bra(layer: LoraLayer, r_rank: int):
        """
        Replace the default LoRA update  (ΔW = B @ A)
        with               ΔW = B @ R @ A   where  R ∈ ℝ^{r×r}.
        """
        # ① create a fresh R parameter, init as identity (or uniform)
        R = torch.nn.Parameter(torch.eye(r_rank, dtype=layer.lora_A["default"].weight.dtype,
                                        device=layer.lora_A["default"].weight.device))
        layer.register_parameter("lora_R", R)

        # ② stash original forward; then monkey-patch
        orig_forward = layer.forward

        def bra_forward(self, x: torch.Tensor):
            # self.lora_A/B are {adapter: Linear}; grab active one
            adapter = getattr(self, "active_adapter", "default")
            A  = self.lora_A[adapter].weight      # (r × in)
            B  = self.lora_B[adapter].weight      # (out × r)
            Rm = self.lora_R                      # (r × r) trainable
            delta_w = (B @ Rm @ A).to(x.dtype)

            return orig_forward(x) + self.scaling * torch.nn.functional.linear(x, delta_w)

        layer.forward = bra_forward.__get__(layer, LoraLayer)  # bind method

    for m in model.modules():
        if isinstance(m, LoraLayer):
            convert_to_bra(m, cfg.rank)
    print("LoRA layers converted to BRA")
    
    model = get_peft_model(model, lora_cfg)

    init_lora_weights(model, cfg.init_variant)
    model.print_trainable_parameters()

    # ---- Dataset ----
    raw_ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    train_ds = safe_slice(raw_ds["train"].shuffle(seed=42), cfg.train_samples)
    val_ds   = safe_slice(raw_ds["validation"].shuffle(seed=42), cfg.val_samples)

    def tok_map(ex):
        prompt = build_prompt(ex)
        ids = tokenizer(prompt, truncation=True, max_length=cfg.max_seq_len)
        ex["input_ids"] = ids["input_ids"]
        return ex

    train_ds = train_ds.map(tok_map, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tok_map, remove_columns=val_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ---- Trainer ----
    targs = TrainingArguments(
        output_dir=f"runs/{cfg.run_name}",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_epochs,
        bf16=cfg.bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=200,
        run_name=cfg.run_name,
    )

    print("Training arguments:")
    print(targs)

    # ---- W&B logging ----
    wandb.init(
        project="lora_ba_init",
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "dataset_name": cfg.dataset_name,
            "dataset_config": cfg.dataset_config,
            "train_samples": cfg.train_samples,
            "val_samples": cfg.val_samples,
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "init_variant": cfg.init_variant,
        },
        tags=[cfg.init_variant],
    )

    model = model.to(device)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("\n**** Starting training ****\n")
    trainer.train()
    wandb.finish()
    print("\n**** Done ****\n")


# --------------------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def parse_cli() -> Config:
    p = argparse.ArgumentParser(description="LoRA BA-initialisation experiment runner")
    p.add_argument("--base_model", type=str, default=Config.base_model)
    p.add_argument("--init_variant", choices=["B_uniform", "A_uniform"], default=Config.init_variant)
    p.add_argument("--run_name", type=str, default=Config.run_name)
    p.add_argument("--train_samples", type=int, default=Config.train_samples)
    p.add_argument("--val_samples", type=int, default=Config.val_samples)
    p.add_argument("--rank", type=int, default=Config.rank)
    p.add_argument("--alpha", type=int, default=Config.alpha)
    p.add_argument("--target_modules", nargs="*", default=None)
    args = p.parse_args()

    cfg_dict = vars(Config())  # default values
    cfg_dict.update(vars(args))
    return Config(**cfg_dict)


if __name__ == "__main__":
    cfg = parse_cli()
    run(cfg)

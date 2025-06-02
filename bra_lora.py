"""Minimal yet extensible script to benchmark different LoRA BA initialisations
on an RTX-class single-GPU box.

Supported variants so far
------------------------
* B_uniform  : B ~ U(-0.01, 0.01),  A = 0  (paper 2406.08447 vanilla)
* A_uniform  : A ~ U(-0.01, 0.01),  B = 0  (paper hypothesis)

The rest of the pipeline (model, dataset, trainer) is identical, so any
performance gap is attributable to the initialisation scheme.

Tested on:
* CUDA 12.1 + RTX 2070 8 GB

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
import time
import numpy as np

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import evaluate
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
    dataset_config: str = "sst2"
    train_samples: int = 20000
    val_samples: int = 2000
    max_seq_len: int = 128
    rank: int = 8
    alpha: int = 16
    batch_size: int = 1
    grad_accum: int = 16
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
        torch.cuda.reset_peak_memory_stats()
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

def build_prompt_sst2(example):
    """Turn an SST-2 sentence into an instruction-tuning prompt."""
    sentence = example["sentence"]
    label = example["label"]
    sentiment = "positive" if label == 1 else "negative"
    prompt = f"Classify the sentiment of the following sentence.\n\nSentence: \"{sentence}\"\nSentiment: {sentiment}\n"
    return prompt

def build_prompt(example, dataset_name):
    """Build a prompt based on the dataset name."""
    if dataset_name == "glue":
        return build_prompt_sst2(example)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name} with config {example.get('dataset_config')}")

# --------------------------------------------------------------------------------------
# Collator that works for both LM‑style token‑level labels and scalar class labels
# --------------------------------------------------------------------------------------
class DualTaskCollator(DataCollatorWithPadding):
    """
    Extends `DataCollatorWithPadding` so it can cope with the two kinds of
    batches produced in this script:
        • Training batches – `labels` is a list[int] (language‑model target)
        • Eval batches     – `labels` is an int       (sentiment class id)

    For language‑model targets we right‑pad the label sequence to the padded
    input length and replace the padding positions with -100 so they are
    ignored by `nn.CrossEntropyLoss`.

    Scalar labels already come out of the super‑call as a 1‑D tensor and need
    no further work.
    """
    def __call__(self, features):
        is_lm_batch = isinstance(features[0]["labels"], list)

        if not is_lm_batch:
            # Simple classification batch – let the parent handle it.
            return super().__call__(features)

        # ---------- LM batch ----------
        # 1. Detach labels so the parent collator doesn’t choke on them.
        label_lists = [feat.pop("labels") for feat in features]

        # 2. Let the parent collate & pad everything else.
        batch = super().__call__(features)

        # 3. Pad labels to max sequence length and add to the batch.
        max_len = batch["input_ids"].shape[1]
        padded = [
            lbl + [-100] * (max_len - len(lbl))
            for lbl in label_lists
        ]
        batch["labels"] = torch.tensor(padded, dtype=torch.long)

        return batch

# --------------------------------------------------------------------------------------
# Main training pipeline ------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def sst2_prompt_no_answer(sentence: str) -> str:
    return f'Classify the sentiment of this sentence:\n\n"{sentence}"\nSentiment:'

def sst2_prompt_with_answer(sentence: str, label: int) -> str:
    sent = "positive" if label == 1 else "negative"
    return f'{sst2_prompt_no_answer(sentence)} {sent}'

def tok_train(ex, tokenizer):
    prompt = sst2_prompt_with_answer(ex["sentence"], ex["label"])
    ids = tokenizer(prompt, truncation=True, max_length=cfg.max_seq_len).input_ids
    ex["input_ids"] = ids
    ex["labels"]    = ids.copy()                # LM objective
    # ex["labels"] = [-100] * (len(ids) - 1) + [ids[-1]]
    return ex

def tok_eval(ex, tokenizer):
    prompt = sst2_prompt_no_answer(ex["sentence"])
    ex["input_ids"] = tokenizer(prompt, truncation=True, max_length=cfg.max_seq_len).input_ids
    ex["labels"]    = ex["label"]              # 0/1 numeric
    return ex

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
        torch_dtype=torch.bfloat16,
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
            # self.lora_A/B are {adapter: Linear}. `active_adapter` can be a
            # string or a list (e.g. ['default']).  Convert to a single name.
            adapter = getattr(self, "active_adapter", "default")
            if isinstance(adapter, (list, tuple)):
                if len(adapter) != 1:
                    raise RuntimeError(
                        f"BRA forward expects exactly one active adapter, got {adapter}"
                    )
                adapter = adapter[0]
            A  = self.lora_A[adapter].weight      # (r × in)
            B  = self.lora_B[adapter].weight      # (out × r)
            Rm = self.lora_R                      # (r × r) trainable
            delta_w = (B @ Rm @ A).to(x.dtype)

            # `self.scaling` is a dict {adapter_name: scale} in recent PEFT
            # versions; fall back to float for older releases.
            scale = self.scaling[adapter] if isinstance(self.scaling, dict) else self.scaling
            return orig_forward(x) + scale * torch.nn.functional.linear(x, delta_w)

        layer.forward = bra_forward.__get__(layer, LoraLayer)  # bind method

    model = get_peft_model(model, lora_cfg)

    for m in model.modules():
        if isinstance(m, LoraLayer):
            convert_to_bra(m, cfg.rank)
    print("LoRA layers converted to BRA")

    print(f"Model architecture: {model}")
    # layer = model.base_model.model.model.layers[0].self_attn.q_proj   # pick any
    # for name, param in layer.named_parameters():
    #     print(name, param.shape, param.requires_grad)

    init_lora_weights(model, cfg.init_variant)
    model.print_trainable_parameters()

    # ---- Dataset ----
    raw_ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    train_ds = safe_slice(raw_ds["train"].shuffle(seed=42), cfg.train_samples)
    val_ds   = safe_slice(raw_ds["validation"].shuffle(seed=42), cfg.val_samples)
    print(f"Train dataset sample: {train_ds[0]}")
    print(f"Validation dataset sample: {val_ds[0]}")

    # train_ds = train_ds.map(tok_train, remove_columns=train_ds.column_names)
    # val_ds   = val_ds.map(tok_eval,   remove_columns=val_ds.column_names)

    train_ds = train_ds.map(lambda ex: tok_train(ex, tokenizer), remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(lambda ex: tok_eval(ex, tokenizer), remove_columns=val_ds.column_names)

    print(f"Train dataset tokenized sample: {train_ds[0]}")
    print(f"Validation dataset tokenized sample: {val_ds[0]}")

    collator = DualTaskCollator(tokenizer, padding=True)
    # collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ---- Trainer ----
    pos_tok = tokenizer(" positive", add_special_tokens=False).input_ids[1]
    neg_tok = tokenizer(" negative", add_special_tokens=False).input_ids[1]
    print(f"pos_tok: {pos_tok}, neg_tok: {neg_tok}")

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred          # both are np.ndarray
    #     preds = (logits[:, pos_tok] > logits[:, neg_tok]).astype(int)
    #     return acc_metric.compute(predictions=preds, references=labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred                      # both np.ndarray
        preds = (logits[:, pos_tok] > logits[:, neg_tok]).astype(int)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1":       f1_metric.compute(predictions=preds, references=labels)["f1"],
        }

    # sanity check if the train/val samples pass thourgh the model successfully
    with torch.no_grad():
        train_sample = torch.tensor(train_ds[0]["input_ids"]).unsqueeze(0).to(device)
        print(f"train_sample shape: {train_sample.shape}")
        train_labels = torch.tensor(train_ds[0]["labels"]).unsqueeze(0).to(device)
        print(f"train_labels shape: {train_labels.shape}")
        train_pred = model(input_ids=train_sample, labels=train_labels)
        print(f"Train sample prediction: {train_pred.logits.shape}")

        val_sample = torch.tensor(val_ds[0]["input_ids"]).unsqueeze(0).to(device)
        print(f"Validation sample input shape: {val_sample.shape}")
        val_labels = torch.tensor(val_ds[0]["labels"]).unsqueeze(0).to(device)
        print(f"Validation sample labels shape: {val_labels.shape}")
        val_pred = model(input_ids=val_sample)
        print(f"Validation sample prediction: {val_pred.logits.shape}")

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

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    param_percent = (trainable_params / total_params) * 100
    print(f"Trainable params: {trainable_params:.2f} M "
      f"({param_percent:.2f} %)  |  Total: {total_params:.2f} M")
    print(f"LoRA rank: {cfg.rank}, alpha: {cfg.alpha}, init variant: {cfg.init_variant}")

    # ---- W&B logging ----
    # wandb.init(
    #     project="lora_ba_init",
    #     name=cfg.run_name,
    #     config={
    #         **vars(cfg),
    #         "params/total": total_params,
    #         "params/trainable": trainable_params,
    #         "params/percent": param_percent,
    #     },
    #     tags=[cfg.init_variant],
    # )

    model = model.to(device)

    class SST2Trainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only=False, **kwargs):
            # If labels are scalar (sentiment), run custom eval logic
            if inputs["labels"].ndim == 1:
                ids = inputs["input_ids"].to(model.device)
                with torch.no_grad():
                    out = model(input_ids=ids)
                token_logits = out.logits[:, -1, :].cpu()
                return (None, token_logits, inputs["labels"].cpu())
            # otherwise fall back to default (training batches)
            return super().prediction_step(model, inputs, prediction_loss_only, **kwargs)
        
    # sanity check
    train_sample = train_ds[0]
    val_sample = val_ds[0]
    print(f"Train sample: {train_sample}")
    print(f"Validation sample: {val_sample}")

    # class SpeedMemCallback(TrainerCallback):
    #     """
    #     Logs runtime, throughput, and GPU-peak memory *per epoch* and again at the
    #     very end of training.

    #     • on_epoch_end -> epoch/runtime_s, epoch/peak_mem_mb, epoch/samples_per_sec
    #     • on_train_end -> time/train_runtime_s (total wall-time)
    #     """
    #     def on_train_begin(self, args, state, control, **kwargs):
    #         self.t0 = time.time()
    #         self.epoch_t0 = self.t0
    #         if torch.cuda.is_available():
    #             torch.cuda.reset_peak_memory_stats()

    #     def on_epoch_end(self, args, state, control, **kwargs):
    #         epoch_time = time.time() - self.epoch_t0
    #         self.epoch_t0 = time.time()            # reset timer for next epoch

    #         peak_mb = (torch.cuda.max_memory_allocated() / 2**20
    #                 if torch.cuda.is_available() else 0)
    #         # samples_per_sec = state.train_samples_per_second
    #         import pdb; pdb.set_trace()
    #         # samples_per_sec = state.metrics["train_samples_per_second"]
    #         samples_per_sec = 0

    #         wandb.log({
    #             "epoch/runtime_s":       epoch_time,
    #             "epoch/samples_per_sec": samples_per_sec,
    #             "epoch/peak_mem_mb":     peak_mb,
    #         }, step=state.global_step)

    #         if torch.cuda.is_available():
    #             torch.cuda.reset_peak_memory_stats()   # fresh peak for next epoch

    #     def on_train_end(self, args, state, control, **kwargs):
    #         total_runtime = time.time() - self.t0
    #         wandb.log({
    #             "time/train_runtime_s": total_runtime,
    #         }, commit=False)

    class SpeedMemCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called every time the trainer logs something."""
            if logs is None:
                return

            # Throughput is already in the log dict
            samples_per_sec = logs.get("train_samples_per_second")

            # GPU-peak memory since last reset
            peak_mb = (torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available() else 0)

            wandb.log({
                "runtime/step":          logs.get("step", state.global_step),
                "train_samples_per_sec": samples_per_sec,
                "gpu_peak_mem_mb":       peak_mb,
            }, step=state.global_step)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()   # start fresh for next interval

        def on_train_end(self, args, state, control, **kwargs):
            total_runtime = time.time() - self.t0
            wandb.log({"time/train_runtime_s": total_runtime}, commit=False)

    trainer = SST2Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[SpeedMemCallback()],
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

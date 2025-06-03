# better-call-lora

## Using LoRA+

To experiment with LoRA+ we can use the code from the official repository:

```bash
git clone https://github.com/nikhil-ghosh-berkeley/loraplus.git
```

However, most likely due to a version mismatch for the `transformers` package, there will be issues with the `Trainer` class that is called from within the LoRA+ code. What they did is that they passed values as positional arguments to the `Trainer` constructor, but the signature changed in newer versions. This is why we need to update `loraplus/lora_plus.py` by using keyword arguments. Changes start from line 175 (as of 1st of June, 2025, commit a0c44bfec0eff97b9574469366bc6b0ccc28838d):

```python
super().__init__(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    model_init=model_init,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
    optimizers=optimizers,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)
```

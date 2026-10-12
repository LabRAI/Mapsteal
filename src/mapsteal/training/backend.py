"""Fresh-LoRA training backend of the formal experiments (`fresh_backend_automl`: Hugging Face Trainer with per-device batch 1, accumulation 8, optional scheduler/warmup arguments), code unchanged except:
package-relative imports, and removal of the optional DS1000_EXTRA_SAVE_STEPS checkpoint callback (off by default in every formal run). Response-only loss (prompt tokens masked), bf16 on CUDA,
fresh LoRA from the pristine Student (no warm start), HF Trainer, save_strategy=epoch, single-GPU assertions."""
def fresh_backend_automl(rows, manifest, run_dir, recipe, lr_scheduler_type="linear", warmup_ratio=0.0):
    import hashlib
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    from ..gpu_isolation import assert_trainer_is_single_gpu, device_provenance, is_cuda_device
    from .recipe import format_prompt as _format_prompt

    device_block = device_provenance(manifest["device"])
    manifest["device_provenance"] = device_block
    if is_cuda_device(manifest["device"]) and manifest["device"] != "cuda:0":
        raise RuntimeError(f"CUDA training must run on the isolated cuda:0, not {manifest['device']!r}")

    set_seed(manifest["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["base_model"], revision=manifest["base_revision"],
        trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(manifest["device"]).startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        manifest["base_model"], revision=manifest["base_revision"],
        torch_dtype=dtype, trust_remote_code=True, local_files_only=True,
    ).to(manifest["device"])

    set_seed(manifest["seed"])
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=recipe.lora_r, lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout, target_modules=list(recipe.target_modules),
    ))
    if model is base:
        raise RuntimeError("get_peft_model returned the bare base -- no LoRA was attached")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n in trainable):
        raise RuntimeError(f"non-LoRA parameters are trainable: {trainable[:5]}")

    hasher = hashlib.sha256()
    for name in sorted(trainable):
        tensor = dict(model.named_parameters())[name].detach().to("cpu", torch.float32).contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    initial_adapter_sha256 = hasher.hexdigest()

    if recipe.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    class Dataset:
        def __init__(self):
            self.items = []
            for row in rows:
                prompt = _format_prompt(row)
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                full = tokenizer(prompt + row["output"] + tokenizer.eos_token,
                                 add_special_tokens=False)["input_ids"]
                ids = full[:recipe.max_seq_length]
                labels = list(ids)
                masked = min(len(prompt_ids), len(labels))
                labels[:masked] = [-100] * masked
                self.items.append({"input_ids": ids, "labels": labels})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def collate(batch):
        ids = [torch.tensor(r["input_ids"]) for r in batch]
        labels = [torch.tensor(r["labels"]) for r in batch]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [torch.ones_like(x) for x in ids], batch_first=True),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"), logging_dir=str(run_dir / "logs"),
        num_train_epochs=recipe.num_train_epochs, learning_rate=recipe.learning_rate,
        per_device_train_batch_size=recipe.per_device_train_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        bf16=str(manifest["device"]).startswith("cuda"), fp16=False,
        save_strategy="epoch", logging_steps=1, report_to=[],
        remove_unused_columns=False, seed=manifest["seed"], data_seed=manifest["seed"],
        gradient_checkpointing=recipe.gradient_checkpointing,
        lr_scheduler_type=lr_scheduler_type, warmup_ratio=warmup_ratio,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Dataset(), data_collator=collate)
    assert_trainer_is_single_gpu(args, trainer.model)
    trained = trainer.train()
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {k: v for k, v in trained.metrics.items() if isinstance(v, (str, int, float, bool))}
    return {"metrics": metrics, "device_provenance": device_block,
           "observed_optimizer_steps": int(trainer.state.global_step),
           "trainable_parameter_tensors": len(trainable),
           "initial_adapter_sha256": initial_adapter_sha256,
           "fresh_init_verified": True, "warm_started": False}

# ==============================================================================
# QLoRA fine-tuning: LoRA config, TrainingArguments and Trainer assembly.
# All hyperparameters come from config.yaml; W&B logging is optional.
# ==============================================================================

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DefaultDataCollator,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import LoraSettings, TrainingSettings


def build_lora_config(lora: LoraSettings) -> LoraConfig:
    """LoRA configuration for the Qwen model architecture.

    Adapters are applied to the attention projections (q/k/v/o) and to the
    FFN projections (gate/up/down).
    """
    return LoraConfig(
        r=lora.r,                        # rank of the added low-rank matrices
        lora_alpha=lora.alpha,           # generally 2*r
        target_modules=list(lora.target_modules),
        lora_dropout=lora.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def prepare_lora_model(model, lora: LoraSettings):
    """Prepares the quantized model for k-bit training and wraps it with LoRA."""
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(lora))
    return model


def build_training_arguments(training: TrainingSettings, use_wandb: bool = False) -> TrainingArguments:
    return TrainingArguments(
        output_dir=training.output_dir,
        num_train_epochs=training.num_train_epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,  # effective batch = batch_size * grad_accum
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        eval_accumulation_steps=training.eval_accumulation_steps,
        warmup_steps=training.warmup_steps,
        learning_rate=training.learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type=training.lr_scheduler_type,
        weight_decay=training.weight_decay,
        fp16=True,
        logging_steps=training.logging_steps,
        eval_strategy="steps",
        eval_steps=training.eval_steps,
        save_steps=training.save_steps,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,          # lower loss is better
        gradient_checkpointing=True,
        max_grad_norm=training.max_grad_norm,
        disable_tqdm=False,
        report_to=["wandb"] if use_wandb else [],
        label_names=["labels"],
    )


def build_trainer(model, training: TrainingSettings, train_dataset, eval_dataset,
                  use_wandb: bool = False) -> Trainer:
    """Assembles the Trainer with early stopping on eval loss."""
    trainer = Trainer(
        model=model,
        args=build_training_arguments(training, use_wandb=use_wandb),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
    )
    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=training.early_stopping_patience,
        early_stopping_threshold=training.early_stopping_threshold,
    ))
    return trainer

"""Selected Training Configuration (the Student-adaptation recipe of Table 2): fresh LoRA, lr 3e-4, 2 epochs, rank 16 / alpha 32 / dropout 0.05, MLP-only targets (gate/up/down),
linear schedule, warmup 0, effective batch 8 (1 x 8 accumulation), max sequence length 1024, gradient checkpointing, bf16, response-only loss. Seeds: 42/43/44."""
from dataclasses import dataclass

@dataclass(frozen=True)
class LoraRecipe:   # LoRA training recipe fields
    max_seq_length: int = 1024
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

MLP = ('gate_proj', 'up_proj', 'down_proj')
SELECTED_CONFIG = LoraRecipe(learning_rate=3e-4, num_train_epochs=2.0, lora_r=16, lora_alpha=32, lora_dropout=0.05, target_modules=MLP)
LR_SCHEDULER, WARMUP = 'linear', 0.0

def format_prompt(row):   # prompt format of the training rows
    return row.get("prompt") or f"### Instruction\n{row['instruction']}\n\n### Input\n{row['input']}\n\n### Response\n"

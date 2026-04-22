import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HOME"] = "/data/aneesh/models"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unsloth
import torch
import numpy as np
import shutil
import random

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from unsloth import FastLanguageModel


class CFG:
    r = 256
    train_batch_size = 10
    eval_batch_size = 2
    epochs = 1
    lr = 2e-4
    eval_split_ratio = 0.05
    name = "20b_tir"


model_name = "openai/gpt-oss-20b"
dtype = None
max_seq_length = 70_000

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=dtype,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
    device_map="cuda",
)

peft_model = FastLanguageModel.get_peft_model(
    base_model,
    r=CFG.r,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=CFG.r,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

try:
    shutil.rmtree("./unsloth_compiled_cache")
except Exception:
    pass

weights_save_dir = f"./ft/weights/{CFG.name}"
log_save_dir = f"./ft/logs/{CFG.name}"

os.makedirs(weights_save_dir, exist_ok=True)
os.makedirs(log_save_dir, exist_ok=True)

data_path = "/data/aneesh/aime/traindata_v2.json"
print(f"Loading data from {data_path}...")

with open(data_path, "r") as f:
    data1 = json.load(f)

data = []
for x in data1:
    if len(x) < 60_000:
        data.append(x)

random.shuffle(data)

num_eval = int(len(data) * CFG.eval_split_ratio)
full_train = data[:-num_eval]
full_eval = data[-num_eval:]

print(f"Total samples: {len(data)} | Train: {len(full_train)} | Eval: {len(full_eval)}")


class TextDataset(Dataset):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        text = self.data_list[idx]
        tokens = self.tokenizer.encode(text)
        labels = tokens.copy()

        assistant_start_idx = None
        for i in range(1, len(tokens) - 1):
            if (
                tokens[i - 1] == 200007
                and tokens[i] == 200006
                and tokens[i + 1] == 173781
            ):
                assistant_start_idx = i + 1
                break

        if assistant_start_idx is not None:
            labels[: assistant_start_idx + 1] = [-100] * (assistant_start_idx + 1)

        prev, curr, nex, last = 200012, 200006, 29010, 200007
        start_idx = None

        for i in range(1, len(tokens) - 1):
            if tokens[i - 1] == prev and tokens[i] == curr and tokens[i + 1] == nex:
                start_idx = i - 1

            if tokens[i] == last and start_idx is not None:
                labels[start_idx : i + 1] = [-100] * (i + 1 - start_idx)
                start_idx = None

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
        }


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        attention_mask = item["attention_mask"]

        padding_length = max_len - len(input_ids)

        input_ids_padded = torch.cat(
            [input_ids, torch.zeros(padding_length, dtype=torch.long)]
        )
        labels_padded = torch.cat(
            [labels, torch.full((padding_length,), -100, dtype=torch.long)]
        )
        attention_mask_padded = torch.cat(
            [attention_mask, torch.zeros(padding_length, dtype=torch.long)]
        )

        input_ids_list.append(input_ids_padded)
        labels_list.append(labels_padded)
        attention_mask_list.append(attention_mask_padded)

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


train_dataset = TextDataset(full_train, tokenizer)
eval_dataset = TextDataset(full_eval, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.train_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=CFG.eval_batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=8,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = CFG.epochs
LEARNING_RATE = CFG.lr

peft_model.to(DEVICE)

optimizer = torch.optim.AdamW(
    peft_model.parameters(),
    lr=LEARNING_RATE,
    eps=1e-10,
)

total_steps = len(train_loader) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)

train_losses = []
val_losses = []
best_val_loss = float("inf")
no_improve = 0

checkpoint_dir = weights_save_dir

print("Starting training...")

global_step = 0
running_train_loss = 0.0
latest_val_loss = None

for epoch in range(NUM_EPOCHS):
    peft_model.train()
    epoch_train_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch in progress_bar:
        global_step += 1

        first_device = peft_model.device
        input_ids = batch["input_ids"].to(first_device)
        attention_mask = batch["attention_mask"].to(first_device)
        labels = batch["labels"].to(first_device)

        optimizer.zero_grad()

        outputs = peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        epoch_train_loss += loss.item()
        running_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if global_step % 5 == 0:
            avg_step_loss = running_train_loss / 5
            train_losses.append(avg_step_loss)
            running_train_loss = 0.0
            progress_bar.set_postfix({"train_loss": f"{avg_step_loss:.4f}"})

    avg_train_loss = epoch_train_loss / len(train_loader)

    print(f"\nEpoch {epoch + 1} Completed")
    print(f"Average Train Loss: {avg_train_loss:.4f}")
    print("Learning Rate:", optimizer.param_groups[0]["lr"])

train_loss_path = os.path.join(log_save_dir, "train_logged_losses.npy")
np.save(train_loss_path, np.asarray(train_losses, dtype=np.float32))

if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
os.makedirs(checkpoint_dir, exist_ok=True)

print("Merging and saving model...")

peft_model.save_pretrained_merged(checkpoint_dir, save_method="mxfp4")
tokenizer.save_pretrained(checkpoint_dir)

print("\nTraining completed!")
print(f"Best model saved to: {checkpoint_dir}")
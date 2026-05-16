import numpy as np
import evaluate
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    get_scheduler,
)
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm


CHECKPOINT = "bert-base-uncased"
OUTPUT_DIR = "test-trainer"
NUM_EPOCHS = 3
BATCH_SIZE = 8


# ── Section 3.2 — Data ────────────────────────────────────────────────────────

def load_data():
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    return raw_datasets, tokenizer


def tokenize_datasets(raw_datasets, tokenizer):
    def tokenize_batch(batch):
        return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)

    return raw_datasets.map(tokenize_batch, batched=True)


def make_dataloaders(tokenized_datasets, data_collator):
    ds = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    train_dl = DataLoader(ds["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
    eval_dl = DataLoader(ds["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator)
    return train_dl, eval_dl


# ── Section 3.3 — Trainer API ─────────────────────────────────────────────────

def compute_metrics(eval_predictions):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_with_trainer(model, tokenized_datasets, data_collator, tokenizer):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


def evaluate_trainer(trainer, tokenized_datasets):
    prediction_output = trainer.predict(tokenized_datasets["validation"])
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    metric = evaluate.load("glue", "mrpc")
    results = metric.compute(predictions=predictions, references=prediction_output.label_ids)
    print("Validation results:", results)

    for i in range(5):
        print(f"\n[{i}] sentence1: {tokenized_datasets['validation'][i]['sentence1']}")
        print(f"    sentence2: {tokenized_datasets['validation'][i]['sentence2']}")
        print(f"    label={tokenized_datasets['validation'][i]['label']}  predicted={predictions[i]}")


# ── Section 3.4 — Custom training loop ───────────────────────────────────────

def train_custom_loop(model, train_dataloader, eval_dataloader):
    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator()
    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    return model, eval_dl


def evaluate_custom_loop(model, eval_dataloader):
    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    results = metric.compute()
    print("Custom loop validation results:", results)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_datasets, tokenizer = load_data()
    print(raw_datasets)

    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Part 1 — Trainer API
    print("\n── Trainer API ──")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    trainer = train_with_trainer(model, tokenized_datasets, data_collator, tokenizer)
    evaluate_trainer(trainer, tokenized_datasets)

    # Part 2 — Custom training loop with Accelerate
    print("\n── Custom loop + Accelerate ──")
    model2 = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    train_dl, eval_dl = make_dataloaders(tokenized_datasets, data_collator)
    model2, eval_dl = train_custom_loop(model2, train_dl, eval_dl)
    evaluate_custom_loop(model2, eval_dl)

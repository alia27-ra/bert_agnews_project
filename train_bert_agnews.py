#!/usr/bin/env python
"""
Fine-tune bert-base-uncased on AG News.
Saves the best model under ./models/bert-agnews
Evaluates with Accuracy and F1 (weighted).
"""

import os
import random
import numpy as np
import torch

from torchtext.datasets import AG_NEWS
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "checkpoints"
MODEL_DIR = os.path.join("models", "bert-agnews")
VAL_SIZE = 0.1
SEED = 42
MAX_LENGTH = 128

def get_texts(batch):
    # Robustly handle different AG News schemas
    if "text" in batch:
        return batch["text"]
    title = batch.get("title", [""] * len(batch["label"]))
    desc = batch.get("description", [""] * len(batch["label"]))
    return [(t or "") + " " + (d or "") for t, d in zip(title, desc)]

def tokenize_fn(batch, tokenizer):
    return tokenizer(get_texts(batch), truncation=True, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

def main():
    set_seed(SEED)
    print("Loading dataset: ag_news")
from torchtext.datasets import AG_NEWS

train_iter = AG_NEWS(split='train')
test_iter = AG_NEWS(split='test')

train_ds = list(train_iter)
test_ds = list(test_iter)
    test_ds = ds["test"]

    print("Creating validation split")
    split = train_ds.train_test_split(test_size=VAL_SIZE, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]

    labels = train_ds.features["label"].names
    num_labels = len(labels)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    print("Labels:", labels)

    print(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print("Tokenizing...")
    tokenized_train = train_ds.map(lambda b: tokenize_fn(b, tokenizer), batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(lambda b: tokenize_fn(b, tokenizer), batched=True, remove_columns=val_ds.column_names)
    tokenized_test = test_ds.map(lambda b: tokenize_fn(b, tokenizer), batched=True, remove_columns=test_ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=tokenized_val)
    print("Validation metrics:", val_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    print(f"Saving model to {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save label names too (optional; id2label already in config.json)
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        import json
        json.dump({"labels": labels}, f, indent=2)

    print("Done. You can now run Streamlit or Gradio apps using this model.")

if __name__ == "__main__":
    main()

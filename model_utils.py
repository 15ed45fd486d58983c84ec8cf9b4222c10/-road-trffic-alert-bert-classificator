import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load

def prepare_datasets(train_texts, train_labels, val_texts, val_labels, tokenizer):
    """Создаёт токенизированные датасеты для обучения и валидации."""
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")

    return train_dataset, val_dataset

def train_model(train_dataset, val_dataset, classes, model_name, output_dir="./results", num_epochs=3):
    """Обучает модель и сохраняет её в указанной директории."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    accuracy_metric = load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Модель сохранена в директории {output_dir}")

    return model, tokenizer

def load_model_and_tokenizer(model_dir, model_name):
    """Загружает сохранённую модель и токенизатор."""
    if not os.path.exists(model_dir):
        raise ValueError(f"Директория {model_dir} не существует.")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict(model, tokenizer, text, classes):
    """Делает предсказание для заданного текста."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).item()
    return classes[predictions]

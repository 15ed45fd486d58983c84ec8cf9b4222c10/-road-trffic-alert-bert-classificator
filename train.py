from dataset import examples

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from evaluate import load


classes = examples.classes
train_texts = examples.train_texts
train_labels = examples.train_labels
val_texts = examples.val_texts
val_labels = examples.val_labels

# ==============================================================================================================================================================================

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

train_dataset = train_dataset.with_format("torch")
val_dataset = val_dataset.with_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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
trainer.save_model("./trained_model")

# Пример предсказания
test_text = "Когда починят светофор на перекрестке?"
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1).item()
predicted_class = classes[predictions]
print("Predicted class:", predicted_class)

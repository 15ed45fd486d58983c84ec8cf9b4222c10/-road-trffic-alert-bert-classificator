import uvicorn
import os
import asyncio
import threading
from queue import Queue
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import examples
from datasets import Dataset
import numpy as np
from evaluate import load
from transformers import Trainer, TrainingArguments


app = FastAPI()

model: Optional[AutoModelForSequenceClassification] = None
tokenizer: Optional[AutoTokenizer] = None
classes = examples.classes

# Флаги состояния
training_in_progress = False
training_complete = False

# Очереди для обработки запросов
queue1 = Queue()
queue2 = Queue()
current_queue = queue1
processing_queue = queue2

# Lock для переключения очередей
queue_lock = threading.Lock()

MODEL_DIR = "./trained_model"
MODEL_NAME = "DeepPavlov/rubert-base-cased"

# Pydantic модель для запроса предсказания
class PredictionRequest(BaseModel):
    text: str


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
    global training_in_progress, training_complete, model, tokenizer

    training_in_progress = True
    training_complete = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))

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

        # Загрузка сохранённой модели
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        training_complete = True
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
    finally:
        training_in_progress = False

def can_launch_backend():
    """Проверяет, чтобы размерности данных совпадали и модель существует."""
    count_text = len(examples.train_texts)
    count_lb = len(examples.train_labels)

    if count_text != count_lb:
        print(f"Размерности данных не совпадают: {count_text} текстов и {count_lb} меток.")
        return False

    if not os.path.exists(MODEL_DIR):
        print(f"Директория модели {MODEL_DIR} не существует.")
        return False

    return True

def load_model():
    """Загружает сохранённую модель и токенизатор."""
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Модель и токенизатор загружены.")

def prediction_worker():
    while True:
        request = processing_queue.get()
        if request is None:
            break  # Завершение работы
        text, response_future = request
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()
            predicted_class = classes[predictions]
            response_future.set_result(predicted_class)
        except Exception as e:
            response_future.set_exception(e)

# Функция переключения очередей
def flip_buffers():
    global current_queue, processing_queue
    with queue_lock:
        current_queue, processing_queue = processing_queue, current_queue

async def process_queue():
    while True:
        flip_buffers()
        while not processing_queue.empty():
            request = processing_queue.get()
            queue1.put(request) if current_queue == queue1 else queue2.put(request)
        await asyncio.sleep(0.1)  # Интервал между переключениями

# Запуск воркера предсказаний
prediction_thread = threading.Thread(target=prediction_worker, daemon=True)
prediction_thread.start()

# Запуск фоновой задачи для обработки очереди
@app.on_event("startup")
async def startup_event():
    global training_in_progress, training_complete, model, tokenizer

    # Проверка размерностей данных
    count_text = len(examples.train_texts)
    count_lb = len(examples.train_labels)
    print(f"Размерности данных: {count_text} текстов и {count_lb} меток.")

    if count_text != count_lb:
        print("Размерности данных не совпадают. Backend не может быть запущен.")
        return

    # Проверка наличия модели
    if not os.path.exists(MODEL_DIR):
        print("Модель не найдена. Начинается обучение модели.")
        tokenizer_temp = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_dataset, val_dataset = prepare_datasets(
            examples.train_texts,
            examples.train_labels,
            examples.val_texts,
            examples.val_labels,
            tokenizer_temp
        )

        training_thread = threading.Thread(
            target=train_model,
            args=(train_dataset, val_dataset, classes, MODEL_NAME, MODEL_DIR, 3),
            daemon=True
        )
        training_thread.start()
    else:
        print("Модель найдена. Загружается модель.")
        load_model()

    # Запуск фоновой задачи для обработки очереди
    asyncio.create_task(process_queue())

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    if training_in_progress:
        raise HTTPException(status_code=503, detail="Модель обучается. Пожалуйста, попробуйте позже.")
    if not training_complete and not model:
        raise HTTPException(status_code=503, detail="Модель не загружена. Пожалуйста, попробуйте позже.")
    
    loop = asyncio.get_event_loop()
    response_future = asyncio.Future()

    # Добавление запроса в очередь
    with queue_lock:
        current_queue.put((request.text, response_future))

    try:
        predicted_class = await asyncio.wait_for(response_future, timeout=10.0)
        return {"predicted_class": predicted_class}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Время ожидания предсказания истекло.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    if training_in_progress:
        return {"status": "Обучение продолжается"}
    elif training_complete:
        return {"status": "Модель готова"}
    elif os.path.exists(MODEL_DIR):
        return {"status": "Модель загружена"}
    else:
        return {"status": "Модель отсутствует"}

# Завершение работы
@app.on_event("shutdown")
def shutdown_event():
    # Остановить воркер предсказаний
    processing_queue.put(None)
    prediction_thread.join()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

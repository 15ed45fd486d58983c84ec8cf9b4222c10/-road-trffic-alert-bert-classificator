import os
from dataset import examples
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def can_launch_backend():
    # Проверяем размеры train_texts и train_labels
    count_text = len(examples.train_texts)
    count_lb = len(examples.train_labels)

    if count_text != count_lb:
        print(f"Размерности данных не совпадают: {count_text} текстов и {count_lb} меток.")
        return False

    print(f"Размерности совпадают: {count_text} текстов и {count_lb} меток.")
    
    # Проверяем существование моделей и токенизаторов
    model_name = "DeepPavlov/rubert-base-cased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Ошибка при загрузке модели или токенизатора: {e}")
        return False

    print("Модель и токенизатор успешно загружены.")
    return True


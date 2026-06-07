import os
import urllib.request
from pathlib import Path

print("APP IMPORTED", flush=True)

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess_input,
)

print("IMPORTS LOADED", flush=True)


# ============================================================
# Диагностика файлов внутри контейнера
# ============================================================

print("CURRENT DIR:", os.getcwd(), flush=True)
print("FILES HERE:", os.listdir("."), flush=True)


# ============================================================
# Пути и URL моделей
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = Path("/tmp/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

KHUSANOV_MODEL_PATH = MODELS_DIR / "khusanov_mobilenetv2_cats_dogs.keras"
RYZHOV_MODEL_PATH = MODELS_DIR / "ryzhov_flowers.keras"

KHUSANOV_MODEL_URL = (
    "https://raw.githubusercontent.com/pashakhu/two-models-classifier/main/"
    "khusanov_mobilenetv2_cats_dogs.keras"
)

RYZHOV_MODEL_URL = (
    "https://raw.githubusercontent.com/pashakhu/two-models-classifier/main/"
    "ryzhov_flowers.keras"
)


def download_file_if_needed(url, path):
    path = Path(path)

    if path.exists() and path.stat().st_size > 1000:
        print(f"MODEL ALREADY EXISTS: {path}", flush=True)
        print(f"SIZE: {path.stat().st_size / 1024 / 1024:.2f} MB", flush=True)
        return

    print(f"DOWNLOADING MODEL FROM: {url}", flush=True)
    print(f"SAVING TO: {path}", flush=True)

    urllib.request.urlretrieve(url, path)

    print(f"DOWNLOADED: {path}", flush=True)
    print(f"SIZE: {path.stat().st_size / 1024 / 1024:.2f} MB", flush=True)


# ============================================================
# Настройки моделей
# ============================================================

MODEL_CONFIGS = {
    "Хусанов — кошки/собаки": {
        "path": str(KHUSANOV_MODEL_PATH),
        "url": KHUSANOV_MODEL_URL,
        "classes": ["Cat", "Dog"],
        "fallback_size": (128, 128),
        "task": "binary",
    },
    "Рыжов — цветы": {
        "path": str(RYZHOV_MODEL_PATH),
        "url": RYZHOV_MODEL_URL,
        "classes": ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
        "fallback_size": (64, 64),
        "task": "multiclass",
    },
}


# ============================================================
# Ленивая загрузка моделей
# ============================================================

loaded_models = {}


def get_model(model_name):
    if model_name not in loaded_models:
        config = MODEL_CONFIGS[model_name]

        model_path = config["path"]
        model_url = config["url"]

        print(f"LOADING MODEL: {model_name}", flush=True)
        print(f"MODEL PATH: {model_path}", flush=True)
        print(f"MODEL URL: {model_url}", flush=True)

        download_file_if_needed(model_url, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        print("START KERAS LOAD_MODEL", flush=True)

        loaded_models[model_name] = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": mobilenet_preprocess_input,
            },
            safe_mode=False,
            compile=False,
        )

        print(f"MODEL LOADED: {model_name}", flush=True)

    return loaded_models[model_name]


def get_model_input_size(model, fallback_size):
    try:
        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        if height is None or width is None:
            return fallback_size

        return int(width), int(height)

    except Exception:
        return fallback_size


# ============================================================
# Предсказание
# ============================================================

def predict(model_name, image):
    print("PREDICT CALLED", flush=True)
    print(f"SELECTED MODEL: {model_name}", flush=True)

    if image is None:
        return {"Загрузите изображение": 1.0}

    config = MODEL_CONFIGS[model_name]

    model = get_model(model_name)

    class_names = config["classes"]
    task = config["task"]

    image_size = get_model_input_size(
        model,
        config["fallback_size"]
    )

    print(f"IMAGE SIZE USED: {image_size}", flush=True)

    image = image.convert("RGB")
    image = image.resize(image_size)

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    predictions = np.array(predictions)

    print(f"RAW PREDICTION SHAPE: {predictions.shape}", flush=True)
    print(f"RAW PREDICTION: {predictions}", flush=True)

    result = {}

    if task == "binary":
        prob = float(predictions[0][0])

        result[class_names[0]] = 1.0 - prob
        result[class_names[1]] = prob

    else:
        probs = predictions[0]
        count = min(len(class_names), len(probs))

        for i in range(count):
            result[class_names[i]] = float(probs[i])

    print(f"RESULT: {result}", flush=True)

    return result


# ============================================================
# Интерфейс Gradio
# ============================================================

print("BUILDING GRADIO INTERFACE", flush=True)

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            value="Хусанов — кошки/собаки",
            label="Выберите модель",
        ),
        gr.Image(
            type="pil",
            label="Загрузите изображение",
        ),
    ],
    outputs=gr.Label(
        label="Результат распознавания",
        num_top_classes=5,
    ),
    title="Классификатор изображений: кошки/собаки и цветы",
    description=(
        "Выберите модель и загрузите изображение. "
        "Модель Хусанова распознаёт кошек и собак. "
        "Модель Рыжова распознаёт виды цветов."
    ),
    examples=[],
)

print("GRADIO INTERFACE READY", flush=True)


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    print("LAUNCHING APP", flush=True)

    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=False,
        debug=True,
    )

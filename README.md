---
title: Two Image Classifiers Mesop App
sdk: docker
app_port: 7860
---

# Two Image Classifiers Mesop App

Веб-приложение на Mesop для выбора одной из двух моделей и классификации загруженной фотографии.

## Файлы моделей

Положите в папку `models/`:

- `khusanov_mobilenetv2_cats_dogs.keras`
- `ryzhov_mobilenetv2_flowers.keras`

## Как сохранить модели из ноутбуков

### Ноутбук Хусанова

После обучения MobileNetV2 fine-tuned:

```python
transfer_model.save('khusanov_mobilenetv2_cats_dogs.keras')
```

### Ноутбук Рыжова

После обучения MobileNetV2 fine-tuned:

```python
model.save('ryzhov_mobilenetv2_flowers.keras')
```

## Запуск локально

```bash
pip install -r requirements.txt
mesop app.py --host 0.0.0.0 --port 7860
```

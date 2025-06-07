# YOLO Object Detection with PyTorch Lightning

## Overview

Этот проект обеспечивает околонадёжный фреймворк для обучения модели YOLO12 с использованием PyTorch Lightning и ее инференса. Реализация использует архитектуру Ultralytics YOLO, предоставляя гибкий конвейер обучения и тестирования с использованием конфигурацией Hydra и интерфейсом командной строки Fire.

## Project Structure

```
project/
├── configs/               # Hydra configuration files
│   ├── model/             # Model configurations
│   ├── data/              # Dataset configurations
│   ├── inference/         # Inference configurations
│   ├── validation/        # Validation configurations
│   └── training/          # Training configurations
├── src/                   # Source code
│   ├── __init__.py
│   ├── model.py           # LightningModule implementations
│   ├── data.py            # LightningDataModule implementations
│   ├── hydra_training.py  # Training entry point
│   ├── unzip_data.py      # For dvc
│   ├── inference.py       # inference configurations
│   └── validation.py      # Validation configurations
├── plots/                 # Results of training
├── runs/                  # Results of validation
├── .flake8                # Resolve issues between black and flake8
├── pyproject.toml         # Dependencies
├── uv.lock                # UV config
├── dvc.yaml               # Data loading
├── commands.py            # CLI entry point
└── README.md
```

## Setup

Используется UV

Код отлажен для Windows 11, CUDA 12.8. Ultralytics крайне капризна при ручном обучении (не используя model.train()), поэтому пришлось сделать много крайне не этичных действий для достижения стабильного обучения используя lightning: правки к лоссу, изменение функциональности некоторых библиотечных функций, отсутствие вычисления метрик при обучении (по какой-то причине вывод модели меняется при обучении, из-за этого летит функция постобработки сырого вывода модели - ничего с этим поделать не могу, метрики можно посчитать уже после обучения) и другие...

### Installation

1. **Создание окружения и установка пакетов**

```bash
uv sync
```

2. **ВАЖНО! Для корректной работы обязательно выполнить следующую команду:**

```bash
uv pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Train

### Получение данных

Можно скачать любой датасет в формате COCO и использовать его в проекте ил же скачать уже подготовленный датасет:

1. Скачивание подготовленного датасета:

```bash
dvc init
```

```bash
dvc repro
```

2. Настройка конфигов

Вообще, при правильной распаковке данных и правильном запуске править ничего не надо, но если очень хочется то конфиги обладают следующей структурой:

```
configs/
├── data/
│   ├── default.yaml - конфиг загрузки датасетов и даталоадеров. Все параметры говорят за себя, можно их менять
│   └── data.yaml - конфиг COCO данных. Можно его заменить на другой
├── model/
│   ├── default.yaml - конфиг загрузки модели. Параметры save_onnx, save_tensorrt отвечают за конвертацию лучшей модели после обучения в соответствующие фреймворки
│   └── yolo12.yaml - конфиг yolo-модели. Можно поменять параметр nc - это количество классов в данных
├── validaion/
│   └── default.yaml - конфиг валидации модели после обучения для подсчета метрик. Нужно указать путь к модели и файлу data.yaml в формате COCO
├── validaion/
│   └── default.yaml - конфиг инференса модели после обучения для визуальной проверки. Нужно указать путь к модели и изображению (поддержки видео формата пока что нет)
└── training/
    └── default.yaml  конфиг обучения модели. Можно изменять разделы trainer и train_hyp
```

**Запуск обучения**

```bash
uv run commands.py train
```

## Production Preparation

В конфиге по умолчанию установлен экспорт модели в форматы onnx и tensorrt. При желании любую yolo модель можно руками переконвертировать в интересующий формат:

1. **ONNX:**

```python
from ultralytics import YOLO

model = YOLO("trained_yolo.pt")
model.export(format="onnx")
```

2. **TensorRT:**

```python
model.export(format="engine", device=0)  # device: GPU index
```

## Inference

После настройки конфигов, достаточно запустить следующую команду:

```bash
uv run commands.py inference
```

**Валидация**

Также можно посчитать метрики обученной модели:

```bash
uv run commands.py val
```

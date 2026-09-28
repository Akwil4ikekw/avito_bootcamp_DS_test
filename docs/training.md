# Повторение обучения

Готовый `artifacts/model/` уже позволяет получить submission без обучения.
Ниже — установка исходников и стадий завершённого эксперимента на Linux с NVIDIA GPU.
Нужны Python 3.12, 64-bit, Git и сохранённые кропы с CSV разбиения.

## Окружение

Создайте отдельное GPU-окружение. Не устанавливайте `paddlepaddle` и
`paddlepaddle-gpu` одновременно. Рабочее окружение использовало сборку CUDA 12.6:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -r requirements-train.txt

git clone https://github.com/PaddlePaddle/PaddleX.git vendor/PaddleX
git -C vendor/PaddleX checkout ffb64904d23708863ff5b8da312a5cbd52a7f462
git clone https://github.com/PaddlePaddle/PaddleClas.git vendor/PaddleX/paddlex/repo_manager/repos/PaddleClas
git -C vendor/PaddleX/paddlex/repo_manager/repos/PaddleClas checkout f1233c18455b8acde4fc42ab0bea575fa06daa8e
python -m pip install --no-deps -e vendor/PaddleX
PYTHONPATH="$PWD/vendor/PaddleX:$PWD/vendor/PaddleX/paddlex/repo_manager/repos/PaddleClas" python -m paddlex --install PaddleClas --use_local_repos --no_deps
```

Штатная регистрация создаёт маркер `.installed`, необходимый PaddleX.
`--no_deps` здесь намеренный: рецепт использует закреплённые зависимости рабочего
окружения, а старый requirements PaddleClas ограничивает другие версии NumPy/OpenCV.
Не добавляйте `--update_repos`: это изменит закреплённые исходники.
Полная чистая установка этим набором команд отдельно не проверялась.
При несовместимом драйвере выберите сборку по
[официальной инструкции Paddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html).

## Данные

Сохранённые данные автор добавляет в
[папку Google Drive](https://drive.google.com/drive/folders/1G5tqSNlCG89dKWfuCzvM4TkXhgPvkCPc?usp=sharing).
Статус архива, ожидаемая структура и проверка доступа — в [data.md](data.md).

Основной путь — использовать сохранённые `data/processed/line_crops/` и CSV сплитов.
Это сохраняет точные картинки и разбиение исходного эксперимента. Подготовка H5
включена как отдельная воспроизводимая процедура; она скачивает около 21 GB исходников:

```bash
python src/prepare_data.py download --config configs/data.yaml
python src/prepare_data.py extract --config configs/data.yaml
python src/prepare_data.py filter --config configs/data.yaml
python src/prepare_data.py split --config configs/data.yaml
```

Эти стадии не нужно запускать, если сохранённый датасет уже есть.
Скрипты не перезаписывают готовые результаты. Пути Windows в CSV поддерживаются
и нормализуются при чтении. Версии Pillow/NumPy/scikit-learn могут влиять на
побитовое воспроизведение PNG/порядка; сохранённые данные предпочтительнее.

## Train / evaluation / export

```bash
python src/build_orientation_dataset.py --root .
python src/train.py check --config configs/finetune.yaml
python src/train.py train --config configs/finetune.yaml --physical-gpu-id 5 --dry-run
# После проверки команды и доступности карты:
python src/train.py train --config configs/finetune.yaml --physical-gpu-id 5
python src/train.py evaluate --config configs/finetune.yaml --physical-gpu-id 5
python src/train.py export --config configs/finetune.yaml --physical-gpu-id 5
```

Готовый финальный checkpoint также лежит в
`artifacts/checkpoint/final_weights.pdparams`. Для его экспорта без нового обучения:

```bash
python src/train.py export --config configs/finetune.yaml --checkpoint artifacts/checkpoint/final_weights.pdparams --stage-output outputs/final_export --physical-gpu-id 5
```

Проверки обучения намеренно требуют manifests и исходные CSV даже для экспорта.
Веса и конфигурация фиксируются в `run.json`, весь вывод — в лог стадии.
Запуск защищён от перезаписи существующей модели. Перед обучением проверяется
порог занятой GPU-памяти (1024 MiB), но это не блокировка карты и не гарантия
отсутствия чужих процессов. Используйте только заранее разрешённую свободную GPU.

Для Brier на размеченной парной validation:

```bash
python src/predict.py --model-dir artifacts/model --device cpu --validation-manifest data/paddlex_textline_orientation/val.txt --validation-root data/paddlex_textline_orientation --validation-metadata data/processed/line_crops/validation.csv --report-dir outputs/validation_reproduced
```

Accuracy PaddleClas и вероятностный Brier — разные проверки. Финальный checkpoint
выбирался по accuracy; Brier считался после обучения. Калибровка на тестовых метках
не выполнялась, поскольку меток нет.

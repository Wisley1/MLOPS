# ArXiv Topic Classifier

Классификатор статей arXiv по 8 темам: art, computer vision, food, games, medicine, microbiome, physics, transformers.

Стек: ClearML (эксперименты, датасет, registry) + DistilBERT + Triton + Streamlit UI.

## Быстрый старт

```bash
pip install -r requirements.txt
pip install clearml-serving

# ClearML Server
cd docker && docker compose up -d && cd ..

# Agent (отдельный терминал)
./scripts/start_agent.sh
```

ClearML UI: http://localhost:8080

Credentials — в `~/clearml.conf` (Settings → Workspace → Create new credentials).

## Пайплайн

**1. Датасет**:

```bash
python -m src.upload_dataset \
  --csv-path arxiv_multiclass_classifier/arxiv_train_multiclass.csv \
  --meta-path arxiv_multiclass_classifier/arxiv_train_multiclass.meta.json \
  --max-rows 7374 \
  --dataset-version 2.0
```



**2. Обучение** (agent вкл):

```bash
export DATASET_ID=<dataset_id>

python -m src.train --remote --queue students \
  --dataset-id "$DATASET_ID" --dataset-version 2.0 \
  --experiment-name exp1-lr2e5-epoch3 \
  --learning-rate 2e-5 --epochs 3 --batch-size 16 --max-samples 0

python -m src.train --remote --queue students \
  --dataset-id "$DATASET_ID" --dataset-version 2.0 \
  --experiment-name exp2-lr5e5-epoch5 \
  --learning-rate 5e-5 --epochs 5 --batch-size 8 --max-samples 0
```

Или оба сразу: `python -m src.enqueue_train --dataset-id <dataset_id>`

**3. Публикация лучшей модели:**

```bash
python -m src.register_model \
  --task-id <BEST_TASK_ID> \
  --model-name arxiv-distilbert \
  --tag production \
  --for-triton
```

В выводе будет `Triton ONNX model published: <id>`.

**4. Serving (Triton + HTTP):**

```bash
CLEARML_TRITON_MODEL_ID=<onnx_model_id> ./scripts/switch_to_triton.sh
./scripts/test_endpoint.sh
```

Serving ID по умолчанию: `84a733bf0b2a4d9f83fb94a89f666c4a` (если создавали другой — передайте аргументом).

**5. UI:**

```bash
export CLEARML_SERVING_URL=http://localhost:8088
streamlit run ui/app.py
```

http://localhost:8501

## Структура

```
src/           upload, train, register, preprocessing
ui/            Streamlit
scripts/       agent, triton, serving, тесты
docker/        ClearML Server
```

## Полезное

- На Mac для Triton нужен Docker Desktop и ~10 GB места под образ
- `--max-samples 0` — обучать на всём датасете
- После `register_model --for-triton` всегда явно указывайте `CLEARML_TRITON_MODEL_ID` при деплое

Ссылки: [репозиторий курса](https://github.com/levkovalenko/ya.camp.2025-clearml) · [ClearML Serving](https://clear.ml/docs/latest/docs/clearml_serving/clearml_serving_cli)

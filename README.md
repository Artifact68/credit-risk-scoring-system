# Credit Risk Scoring System

Полноценный учебный проект кредитного скоринга на данных соревнования
**«Кредитный скоринг — Финтех-школа Tinkoff.ru»**.

Сервис оценивает вероятность дефолта клиента, переводит её в скоринговый балл,
формирует рекомендацию и показывает признаки, которые сильнее всего повлияли
на результат.

## Что входит в проект

- загрузка датасета через `kagglehub`;
- автоматический поиск и объединение таблиц с анкетами и целевой переменной;
- обработка числовых и категориальных признаков;
- интерпретируемая модель логистической регрессии;
- подбор порога решения по F1-score;
- FastAPI с документацией Swagger;
- адаптивный веб-интерфейс;
- объяснение отдельных прогнозов через вклад признаков;
- модульные и API-тесты;
- проверка кода в GitHub Actions;
- запуск в Docker.

## Структура

```text
.
├── app/
│   ├── api/                 # HTTP endpoints
│   ├── core/                # настройки приложения
│   ├── ml/                  # загрузка данных, обучение и инференс
│   ├── main.py              # FastAPI приложение
│   └── schemas.py           # схемы запросов и ответов
├── data/raw/                # исходные CSV-файлы, не хранятся в Git
├── frontend/                # HTML, CSS и JavaScript
├── models/                  # модель и метрики, не хранятся в Git
├── scripts/
│   ├── download_data.py
│   └── train.py
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Быстрый запуск

Требуется Python 3.11 или новее.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### 1. Загрузка данных

Перед первым запуском авторизуйтесь в Kaggle и примите правила соревнования.
После этого выполните:

```bash
python scripts/download_data.py
```

Скрипт использует следующий вызов:

```python
import kagglehub

path = kagglehub.competition_download("fintech-credit-scoring")
print("Path to competition files:", path)
```

Загруженные файлы копируются в `data/raw`.

### 2. Обучение модели

```bash
python scripts/train.py
```

После обучения появятся:

- `models/credit_scoring.joblib` — модель, препроцессинг и схема признаков;
- `models/metrics.json` — ROC-AUC, Average Precision, F1 и выбранный порог.

Скрипт не привязан к жёсткому набору столбцов. Он ищет целевую переменную
по распространённым названиям, находит общий идентификатор и объединяет
таблицы автоматически.

### 3. Запуск приложения

```bash
python -m uvicorn app.main:app --reload
```

Откройте:

- веб-интерфейс: `http://127.0.0.1:8000`;
- Swagger: `http://127.0.0.1:8000/docs`;
- проверка состояния: `http://127.0.0.1:8000/health`.

## API

### `GET /health`

Проверяет работу приложения и наличие обученной модели.

### `GET /api/v1/model`

Возвращает тип модели, метрики и сведения об обучении.

### `GET /api/v1/schema`

Возвращает описание признаков. По этой схеме веб-интерфейс строит форму.

### `POST /api/v1/predict`

Пример запроса:

```json
{
  "features": {
    "income": 85000,
    "employment": "employee"
  }
}
```

Пример ответа:

```json
{
  "default_probability": 0.1842,
  "score": 816,
  "risk_level": "low",
  "decision": "approve",
  "threshold": 0.47,
  "top_factors": []
}
```

Отсутствующие поля заполняются медианами или наиболее частыми значениями,
которые были рассчитаны на обучающей выборке.

## Тесты и проверка стиля

```bash
python -m pytest
python -m ruff check .
```

Или через `Makefile`:

```bash
make test
make lint
```

## Docker

Сначала обучите модель локально, затем запустите:

```bash
docker compose up --build
```

Каталог `models` подключается к контейнеру только для чтения.



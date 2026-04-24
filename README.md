# AI_2_sem_2 -  Proof-of-concept прототип, иллюстрирующий применимость LLM к распознаванию спама.




## Инструкция к запуску

Для развертывания модели труебуется :
-   Запустить ``python3 -m pip install requests pandas scikit-learn tqdm``, дабы вне Dockerа была возможность запускать функции тестирования внешнего подключения к API и составления отчётов ответов модели через соотвтствующие функции в папке ``./scripts``. Может потребовать venv.
-   ``docker compose up -d --build`` для поднятия Docker контейнера.
-   Подождите около 5-10 секунд после начала работы контейнера, перед первым запросом.

  ## Примеры запросов 

  ### 1. Health-check сервиса
``curl -s http://localhost:8000/health | python3 -m json.tool``
### 2. Тестовый запрос через curl(отсылайте через терминал)

``curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Win $1000 now! Click: bit.ly/xyz", "technique": "cot-few-shot"}'``
### 3. Запрос отчёта ответов модели по метрикам на 100 запросов

``python3 scripts/evaluate.py --samples 100```

### 4. Запрос через функции для работы с внешними API

python3 scripts/test_api.py --text "Hey, dinner at 7pm?" --technique cot-few-shot


### Отчёт об запуске LLM на 1000 запросов расположен в 

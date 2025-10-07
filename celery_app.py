from celery import Celery
import os
import sys

# Ensure current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create the Celery app
celery_app = Celery(
    "aiops_tasks",
    broker="amqp://guest:guest@rabbitmq:5672//",
    backend="rpc://"
)

# Explicitly import tasks so Celery can register them
import tasks

celery_app.autodiscover_tasks(["tasks"])

celery_app.conf.task_routes = {
    "tasks.retrain_model_task": {"queue": "retrain_queue"},
}

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

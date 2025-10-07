from celery import Celery

celery_app = Celery(
    "aiops_tasks",
    broker="amqp://guest:guest@localhost:5672//",
    backend="rpc://"
)

celery_app.conf.task_routes = {
    "tasks.retrain_model_task": {"queue": "retrain_queue"},
}

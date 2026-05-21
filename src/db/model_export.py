from sqlalchemy.orm import Session
from src.db import models, schemas

def create_model_metrics(db: Session, metrics: schemas.ModelMetricsCreate):

    db_metrics = models.ModelMetrics(
        model_name=metrics.model_name,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        model_path=metrics.model_path
    )

    db.add(db_metrics)
    db.commit()
    db.refresh(db_metrics)

    return db_metrics


def get_all_models(db: Session):

    return db.query(models.ModelMetrics).all()


def get_best_model(db: Session):

    return db.query(models.ModelMetrics)\
        .order_by(models.ModelMetrics.f1_score.desc())\
        .first()

# NEW FUNCTION
def get_model_by_name(db: Session, model_name: str):

    return db.query(models.ModelMetrics)\
        .filter(models.ModelMetrics.model_name == model_name)\
        .first()


# OPTIONAL UPDATE METHOD (cleaner design)
def update_model_metrics(db: Session, db_model, metrics: schemas.ModelMetricsCreate):

    db_model.accuracy = metrics.accuracy
    db_model.precision = metrics.precision
    db_model.recall = metrics.recall
    db_model.f1_score = metrics.f1_score
    db_model.model_path = metrics.model_path

    db.commit()
    db.refresh(db_model)

    return db_model

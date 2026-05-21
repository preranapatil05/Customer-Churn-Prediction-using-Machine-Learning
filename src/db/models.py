from sqlalchemy import Column, Integer, Float, String
from src.db.database import Base


class ModelMetrics(Base):

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True,autoincrement=True)

    model_name = Column(String, nullable=False)

    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    model_path = Column(String)
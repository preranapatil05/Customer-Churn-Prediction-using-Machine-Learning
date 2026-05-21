from fastapi import APIRouter

from src.components.predict import ChurnPredictor
from src.db.schemas import PredictionRequest, PredictionResponse

router = APIRouter()

# initialize predictor
predictor = ChurnPredictor()


@router.get("/msg")
def home():
    return {"message": "Customer Churn Prediction API"}


@router.post("/predict", response_model=PredictionResponse)
def predict_churn(data: PredictionRequest):

    # convert pydantic object → dictionary
    input_data = data.dict()

    # call predictor
    result = predictor.predict(input_data)

    return result
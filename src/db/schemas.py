from pydantic import BaseModel
class ModelMetricsCreate(BaseModel):

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    model_path: str


class ModelMetricsResponse(ModelMetricsCreate):

    id: int

    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):

    prediction: int
    churn_probability: float
import pickle
import pandas as pd
from src.db.database import SessionLocal
from src.db import model_export
from src.logger import logger


class ChurnPredictor:

    def __init__(self):
        import os

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logger.info("Initializing ChurnPredictor")
        # create DB session
        self.db = SessionLocal()

        # get best model from database
        logger.info("Fetching best model from database")
        best_model = model_export.get_best_model(self.db)

        if best_model is None:
            logger.error("No trained model found in database")
            raise Exception("No trained model found in database")

        # self.model_path = best_model.model_path
        self.model_path = os.path.join(
            BASE_DIR,
            "src",
            "models",
            os.path.basename(best_model.model_path)
        )
        logger.info(f"Resolved model path: {self.model_path}")
        # load model
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully: {best_model.model_name}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        self.db.close()

    def predict(self, input_data: dict):
        logger.info("Received prediction request")
        """
        input_data: dictionary of customer features
        """
        # convert dictionary → DataFrame
        try:
            df = pd.DataFrame([input_data])
            logger.debug(f"Input DataFrame: {df.to_dict()}")
        # make prediction
            prediction = self.model.predict(df)[0]

            probability = self.model.predict_proba(df)[0][1]

            result = {
                "prediction": int(prediction),
                "churn_probability": float(probability)
                }
            logger.info(f"Prediction successful: {result}")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
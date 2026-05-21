# src/data_transformation.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logger


class DataTransformation:

    def __init__(self):
        self.preprocessor = None
        logger.info("DataTransformation class initialized")
    def build_preprocessor(self, X: pd.DataFrame):
        """
        Builds ColumnTransformer:
        - OneHotEncoder categorical features
        - Pass numerical features
        """
        try:
            logger.info("Building preprocessing pipeline")
            categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
            numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

            logger.debug(f"Categorical features: {categorical_features}")
            logger.debug(f"Numerical features: {numerical_features}")

            categorical_transformer = OneHotEncoder(handle_unknown="ignore")

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_transformer, categorical_features),
                    ("num", "passthrough", numerical_features)
                ]
                )
            logger.info("Preprocessor pipeline created successfully")

            return self.preprocessor

        except Exception as e:
            logger.error(f"Error building preprocessor: {e}")
            raise

    def fit_transform(self, X_train: pd.DataFrame):
        """
        Fit on training data and transform
        """
        try:
            logger.info("Training data transformation completed")
            return self.preprocessor.fit_transform(X_train)
        except Exception as e:
            logger.error(f"Error during fit_transform: {e}")
            raise

    def transform(self, X_test: pd.DataFrame):
        """
        Transform test data only
        """
        try:
            return self.preprocessor.transform(X_test)
            logger.info("Test data transformation completed")

        except Exception as e:
            logger.error(f"Error during test data transformation: {e}")
            raise
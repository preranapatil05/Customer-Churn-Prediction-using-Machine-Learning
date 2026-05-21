# src/data_preprocessing.py

import pandas as pd
from src.logger import logger


class DataPreprocessing:

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning:
        - Drop customerID
        - Convert TotalCharges to numeric
        - Handle missing values
        """
        try:
            logger.info("Starting data cleaning process")
            if "customerID" in df.columns:
                logger.debug("Dropping customerID column")
                df = df.drop("customerID", axis=1)

        # Convert TotalCharges to numeric
            logger.debug("Converting TotalCharges to numeric")
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Fill missing values
            logger.debug("Handling missing values in TotalCharges")
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
            logger.info("Data cleaning completed successfully")

            return df
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise

    def separate_features_target(self, df: pd.DataFrame):
        """
        Separate independent and dependent variables
        """
        try:
            logger.info("Separating features and target variable")

            X = df.drop("Churn", axis=1)
            y = df["Churn"].map({"Yes": 1, "No": 0})
            logger.info(f"Feature separation successful | X shape: {X.shape}, y shape: {y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error separating features and target: {e}")
            raise
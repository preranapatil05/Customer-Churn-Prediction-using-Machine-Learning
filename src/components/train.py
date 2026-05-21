import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.components.model_pipeline import ModelPipeline

from src.db.database import engine, SessionLocal
from src.db import models, model_export, schemas
from src.logger import logger
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# create table
models.Base.metadata.create_all(bind=engine)

def train_model():
    try:
        logger.info("Starting model training pipeline")
        # Data Ingestion
        logger.info("Running Data Ingestion")
        ingestion = DataIngestion(os.path.join(BASE_DIR, "src", "data", "Telco.csv"))
        train_df, test_df = ingestion.run()

        # Data Preprocessing
        logger.info("Running Data Preprocessing")
        preprocessing = DataPreprocessing()

        train_df = preprocessing.clean_data(train_df)
        test_df = preprocessing.clean_data(test_df)

        X_train, y_train = preprocessing.separate_features_target(train_df)
        X_test, y_test = preprocessing.separate_features_target(test_df)
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        # Data Transformation
        logger.info("Building preprocessing pipeline")
        transformation = DataTransformation()
        preprocessor = transformation.build_preprocessor(X_train)

        # Model Builder
        model_builder = ModelPipeline(preprocessor)

        # DB session
        logger.info("Creating database session")
        db = SessionLocal()

        # Random Forest Model
        logger.info("Training Random Forest model")
        rf_model = model_builder.build_random_forest_pipeline()

        rf_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)

        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_precision = precision_score(y_test, rf_pred)
        rf_recall = recall_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred)

        print("\n===== Random Forest Results =====")
        print("Accuracy:", rf_accuracy)
        print(classification_report(y_test, rf_pred))

        # save model
        os.makedirs(os.path.join(BASE_DIR, "src", "models"), exist_ok=True)

        rf_model_path = os.path.join(BASE_DIR, "src", "models", "random_forest_model.pkl")
        # rf_model_path = "src/models/random_forest_model.pkl"
        logger.info("Saving Random Forest model")
        #
        # with open(rf_model_path, "wb") as f:
        #     pickle.dump(rf_model, f)

        existing_rf = model_export.get_model_by_name(db, "RandomForest")

        # store metrics in DB
        rf_metrics = schemas.ModelMetricsCreate(
            model_name="RandomForest",
            accuracy=rf_accuracy,
            precision=rf_precision,
            recall=rf_recall,
            f1_score=rf_f1,
            model_path=rf_model_path
        )

        if existing_rf:

            logger.info(f"Existing RF F1: {existing_rf.f1_score}")
            logger.info(f"New RF F1: {rf_f1}")

            if rf_f1 > existing_rf.f1_score:

                logger.info("New RF model improved. Updating model.")

                with open(rf_model_path, "wb") as f:
                    pickle.dump(rf_model, f)

                model_export.update_model_metrics(db, existing_rf, rf_metrics)

            else:

                logger.info("Existing RF model is better. Skipping update.")

        else:

            logger.info("No RF model found. Saving new model.")

            with open(rf_model_path, "wb") as f:
                pickle.dump(rf_model, f)

            model_export.create_model_metrics(db, rf_metrics)
        # logger.info("Random Forest metrics stored in database")


        # XGBoost Model
        logger.info("Training XGBoost model")
        xgb_model = model_builder.build_xgboost_pipeline()

        xgb_model.fit(X_train, y_train)

        xgb_pred = xgb_model.predict(X_test)

        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_precision = precision_score(y_test, xgb_pred)
        xgb_recall = recall_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred)

        print("\n===== XGBoost Results =====")
        print("Accuracy:", xgb_accuracy)
        print(classification_report(y_test, xgb_pred))



        xgb_model_path = os.path.join(BASE_DIR, "src", "models", "xgboost_model.pkl")
        # xgb_model_path = "src/models/xgboost_model.pkl"
        logger.info("Saving XGBoost model")

        existing_xgb = model_export.get_model_by_name(db, "XGBoost")

        # store metrics in DB
        xgb_metrics = schemas.ModelMetricsCreate(
            model_name="XGBoost",
            accuracy=xgb_accuracy,
            precision=xgb_precision,
            recall=xgb_recall,
            f1_score=xgb_f1,
            model_path=xgb_model_path
        )

        if existing_xgb:

            logger.info(f"Existing XGB F1: {existing_xgb.f1_score}")
            logger.info(f"New XGB F1: {xgb_f1}")

            if xgb_f1 > existing_xgb.f1_score:

                logger.info("New XGB model improved. Updating model.")

                with open(xgb_model_path, "wb") as f:
                    pickle.dump(xgb_model, f)

                model_export.update_model_metrics(db, existing_xgb, xgb_metrics)

            else:

                logger.info("Existing XGB model is better. Skipping update.")

        else:

            logger.info("No XGB model found. Saving new model.")

            with open(xgb_model_path, "wb") as f:
                pickle.dump(xgb_model, f)

            model_export.create_model_metrics(db, xgb_metrics)

        db.close()
        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    train_model()
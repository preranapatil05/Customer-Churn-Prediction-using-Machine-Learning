from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.logger import logger

class ModelPipeline:

    def __init__(self, preprocessor):
        """
        Takes the preprocessor (ColumnTransformer) as input
        """
        self.preprocessor = preprocessor
        self.model = None
        logger.info("ModelPipeline initialized with provided preprocessor")

    # Random Forest Pipeline
    def build_random_forest_pipeline(self):
        """
        Combines Preprocessor + RandomForest into a Pipeline
        """
        try:
            logger.info("Building Random Forest pipeline")
            self.model = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        max_depth=None,
                        class_weight="balanced"
                    ))
                ]
            )
            logger.info("Random Forest pipeline created successfully")

            return self.model
        except Exception as e:
            logger.error(f"Error building Random Forest pipeline: {e}")
            raise



    # XGBoost Pipeline
    def build_xgboost_pipeline(self):
        """
        Combines Preprocessor + XGBoost into a Pipeline
        """
        try:
            logger.info("Building XGBoost pipeline")
            self.model = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=2.7,
                        random_state=42,
                        eval_metric="logloss"
                    ))
                ]
            )
            logger.info("XGBoost pipeline created successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error building XGBoost pipeline: {e}")
            raise
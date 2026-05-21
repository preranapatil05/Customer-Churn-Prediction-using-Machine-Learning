import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logger


class DataIngestion:
    def __init__(self,data_path:str):
        self.data_path=data_path
        self.train_path=os.path.join("src","data","train.csv")
        self.test_path=os.path.join("src","data","test.csv")

    def load_data(self) ->pd.DataFrame:
        try:
            logger.info(f"Loading dataset from {self.data_path}")
            df=pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise Exception(f"Error loading data:{e}")

    def split_data(self,df:pd.DataFrame,test_size:float=0.2):
        try:
            logger.info("Splitting dataset into train and test sets")
            train_df,test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=42,
                stratify=df['Churn']
            )
            logger.info(
                f"Data split successful | Train shape: {train_df.shape} | Test shape: {test_df.shape}"
            )
            return train_df,test_df
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise Exception(f"Error splitting data:{e}")

    def save_data(self,train_df:pd.DataFrame,test_df:pd.DataFrame):
        try:
            logger.info("Saving train and test datasets")
            train_df.to_csv(self.train_path,index=False)
            test_df.to_csv(self.test_path,index=False)
            logger.info("Train and Test data saved successfully")

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise Exception(f"Error saving data: {e}")

    def run(self):
        logger.info("Starting Data Ingestion Pipeline")
        df=self.load_data()
        train_df,test_df=self.split_data(df)
        self.save_data(train_df,test_df)
        logger.info("Data Ingestion Pipeline Completed")
        return train_df,test_df

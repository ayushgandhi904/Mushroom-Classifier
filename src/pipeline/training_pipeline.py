import os, sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object


#Code to connect pipeline

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)

    
    

    
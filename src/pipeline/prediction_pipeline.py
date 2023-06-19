import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import numpy as np
import pandas as pd

class PredictPipeline:
    
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            
            logging.info("Starting for prediction path")
            
            preprocessor_path = os.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            logging.info("Model able to predict")
            
            return pred
        
        except Exception as e:
            logging.info("Exception in prediction step")
            raise CustomException(e, sys)


            
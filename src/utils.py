import os, sys, pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok= True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("File dump as pickle file")
        
    except Exception as e:
        logging.info("File not able to dump as pickle")
        raise CustomException(e, sys)
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        logging.info("Creating model report")
        for i in range(len(models)):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)
            
            #Predicting value
            y_test_pred = model.predict(X_test)
            
            #getting accuracy score
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            logging.info("Report generated")
            
        return report
    
    except Exception as e:
        logging.info("Exception as model training step")
        raise CustomException(e, sys)
    
def load_object(file_path):
    
    try:
        logging.info("Trying to open file_path")
        
        with open(file_path, "rb") as file_obj:
            logging.info("Pickle file loaded")
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info("Exception in utils load object")
        raise CustomException(e, sys)
    
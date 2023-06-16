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
    
    
    
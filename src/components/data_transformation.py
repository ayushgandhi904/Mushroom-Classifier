import os, sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    
    
    def get_data_transformation(self):
        self.get_data_transformation_config = DataTransformationConfig()
        
    def get_data_tranformation_object(self):
        try:
            logging.info("Data Transformation step started")
            
        except Exception as e:
            logging.info("Error occur in Data transformation stage")
            raise CustomException(e, sys)
        
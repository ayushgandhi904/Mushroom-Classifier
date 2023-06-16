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
from sklearn.compose import ColumnTransformer, make_column_transformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    
class DataTransformation:
    
    def __init__(self):
        self.get_data_transformation_config = DataTransformationConfig()
        
    def get_data_tranformation_object(self):
        try:
            logging.info("Data Transformation step started")
            
            lab_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                        'stalk-surface-below-ring', 'stalk-color-above-ring',
                        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                        'ring-type', 'spore-print-color', 'population', 'habitat']
            
            logging.info("Pipeline initiated")
            
            target_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("labelencoder", LabelEncoder()),
                    ("PCA", PCA(n_components=10))
                ]
            )
            
            logging.info("Processor step started")
            
            preprocessor = ColumnTransformer([
                "lab_pipeline", target_pipeline, lab_cols
            ])
            logging.info("Pipeline completed")
            
            return preprocessor
                        
        except Exception as e:
            logging.info("Error occur in Data transformation stage")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            logging.info("Steps started for test & train")
            
        except Exception as e:
            logging.info("Error occur in datatransformation intialization")
            raise CustomException(e, sys)
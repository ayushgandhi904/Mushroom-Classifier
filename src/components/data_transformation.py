import os, sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

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
                    ("ordinalencoder", OrdinalEncoder()),
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
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train & Test data readed")
            
            logging.info(f"Train data is ready: \n {train_df.head(2).to_string()}")
            logging.info(f"Test data is ready: \n {test_df.head(2).to_string()}")  
            
            logging.info("Ready for preprocessing object")
            
            preprocessing_obj = self.get_data_tranformation_object()
            
            target_column = "class"
            drop_columns = [target_column]
            
            #Train set
            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)
            target_feature_train_df = train_df[target_column]
            
            #Test set
            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column]
            logging.info("Target & Features columns sepearted")
            
            #Transforming into preprocessor
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Preprocessing applied to training & test datasets") 
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("File saved in Pickle file")
            
            return (
                train_arr,
                test_arr,
                self.get_data_transformation_config.preprocessor_obj_file_path
            )        
            
        except Exception as e:
            logging.info("Error occur in datatransformation intialization")
            raise CustomException(e, sys)
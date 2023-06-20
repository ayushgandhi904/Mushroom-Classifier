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
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation step started")
            
            lab_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                        'stalk-surface-below-ring', 'stalk-color-above-ring',
                        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                        'ring-type', 'spore-print-color', 'population', 'habitat']
            
            #Ranking for ordinal variable
            
            cap_shape = ['b','c','x','f','k','s']
            cap_surface = ['f','g','y','s']
            cap_color = ['n','b','c','g','r','p','u','e','w','y']
            bruises = ['t','f']
            odor = ['a','l','c','y','f','m','n','p','s']
            gill_attachment = ['a','f']
            gill_spacing = ['c','w']
            gill_size = ['b','n']
            gill_color = ['k','n','b','h','g','r','o','p','u','e','w','y']
            stalk_shape = ['e','t']
            stalk_root = ['b','c','e','r','?']
            stalk_surface_above_ring = ['f','y','k','s']
            stalk_surface_below_ring = ['f','y','k','s']
            stalk_color_above_ring = ['n','b','c','g','o','p','e','w','y']
            stalk_color_below_ring = ['n','b','c','g','o','p','e','w','y']
            veil_type = ['p']
            veil_color = ['w','n','o','y']
            ring_number = ['n','o','t']
            ring_type = ['e','f','l','n','p']
            spore_print_color = ['k','n','b','h','r','o','u','w','y']
            population = ['a','c','n','s','v','y']
            habitat = ['g','l','m','p','u','w','d']
            
            logging.info("Pipeline initiated")
            
            target_pipeline = Pipeline(
                steps = [
                    ("ordinalencoder", OrdinalEncoder(categories=[cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, 
                                                                  gill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring,
                                                                  stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type,
                                                                  veil_color,ring_number, ring_type, spore_print_color, population, habitat])),
                                                                  
                    ("PCA", PCA(n_components=10))
                ]
            )
            
            logging.info("Processor step started")
            
            preprocessor = ColumnTransformer([
                ("lab_pipeline", target_pipeline, lab_cols)
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
            
            preprocessing_obj = self.get_data_transformation_object()
            
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
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("File saved in Pickle file")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )        
            
        except Exception as e:
            logging.info("Error occur in datatransformation intialization")
            raise CustomException(e, sys)
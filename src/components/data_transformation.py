import os, sys #for system
import numpy as np #for array reading
import pandas as pd #handling dataframe
from src.logger import logging #to log the file
from src.exception import CustomException #to raise error as exception
from dataclasses import dataclass #directly initializing the class
from src.utils import save_object #function of saving the object
from sklearn.preprocessing import OrdinalEncoder #for data transformation
from sklearn.decomposition import PCA #for dimension reduction of data
from sklearn.pipeline import Pipeline #Creating pipeline structure
from sklearn.compose import ColumnTransformer #For transforming all columns of pipeline

#Data class for Data Transformation --> For creating preprocessor file for transorming data
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl") #path for joining preprocessor file
    logging.info("Dataclass of Data Transformation able to run sucessfully")
    
#Class for Data Transformation
class DataTransformation:
    def __init__(self): #Initializing with DataTransformation config
        self.data_transformation_config = DataTransformationConfig()
        logging.info("Data Transformation initialized")
    
    #Defining function for Transforming the data    
    def get_data_transformation_object(self):
        logging.info("Data Transfomation method started")
        try:            
            lab_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                        'stalk-surface-below-ring', 'stalk-color-above-ring',
                        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                        'ring-type', 'spore-print-color', 'population', 'habitat'] #columns to apply the pipeline
            
            #Giving Label for each feature of column            
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
            
            logging.info("Label initiated for each column")
            
            #creating pipeline for the define column
            target_pipeline = Pipeline(
                steps = [
                    ("ordinalencoder", OrdinalEncoder(categories=[cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, 
                                                                  gill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring,
                                                                  stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type,
                                                                  veil_color,ring_number, ring_type, spore_print_color, population, habitat])), #ordinal encoding to columns
                                                                  
                    ("PCA", PCA(n_components=10)) #PCA to colums
                ]
            )
            
            logging.info("Targe pipeline created")
            
            #Preprocessor to transform the columns
            preprocessor = ColumnTransformer([
                ("lab_pipeline", target_pipeline, lab_cols)
            ])
            logging.info("Preprocessor object created")
            return preprocessor
        
        #store the exception if occours                 
        except Exception as e:
            logging.info("Error occur in Data transformation stage")
            raise CustomException(e, sys)
    
    #Defining function to apply transformation on train & test data    
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Transformation step on train & test started")
        
        try:
            train_df = pd.read_csv(train_path) #to read train data
            test_df = pd.read_csv(test_path) #to read test data
            
            logging.info("Train & Test data readed")       
            logging.info(f"Train data is ready: \n {train_df.head(2).to_string()}")
            logging.info(f"Test data is ready: \n {test_df.head(2).to_string()}")  
            
            preprocessing_obj = self.get_data_transformation_object() #Creating the preprocessor object
            logging.info("Preprocessor object readed on data")
            
            target_column = "class" #defining targeted coloumn
            drop_columns = [target_column] #dropping targeted coloumn from the set
            logging.info("Target column separated")
            
            
            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1) 
            target_feature_train_df = train_df[target_column]
            logging.info("Train data has separated the target column")
            
            #Defining the Test df by dropping the targeted column
            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column]
            logging.info("Test data has separated the target column")
            
            #Transforming into Preprocessor object to data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) #train data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df) #test data
            logging.info("Preprocessing applied to training & test datasets") 
            
            #Converting into numpy array for train & test data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] #train array 
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] #test array
            logging.info("Data transfom into the array")
            
            #function from utils to save the preprocessor file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Preprocessor file saved as Pickle file")
            
            return (
                train_arr, #train array
                test_arr, #test array
                self.data_transformation_config.preprocessor_obj_file_path
            )        
            
        #store the exception if occours     
        except Exception as e:
            logging.info("Error occur in datatransformation intialization")
            raise CustomException(e, sys)
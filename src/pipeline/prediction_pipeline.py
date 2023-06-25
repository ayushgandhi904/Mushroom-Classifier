import os, sys #for system
from src.exception import CustomException #for the exception if occurs
from src.logger import logging #to log the file
from src.utils import load_object #to load pickle file of preprocessor & model
import pandas as pd #handling the dataframe

#Creating class for prediction pipeline
class PredictPipeline:
    
    def __init__(self): #initializing
        pass    
    
    def predict(self, features):
        try:   
            logging.info("Starting for prediction path")
            
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl") #joining the preprocessor path
            model_path = os.path.join("artifacts", "model.pkl") #joining the model path
            logging.info("Preprocessor & Model path joined")
            
            #loading preprocessor object
            preprocessor = load_object(preprocessor_path) 
            model = load_object(model_path)
            logging.info("Preprocessor object loaded")
            
            #to transform the features through preprocessor
            data_scaled = preprocessor.transform(features) 
            logging.info("Features transform through Preprocessor")
            
            #predicting the output through model 
            pred = model.predict(data_scaled)
            logging.info("Output predicted through model")
            
            return pred
        
        #store the exception if occours  
        except Exception as e:
            logging.info("Exception in prediction step")
            raise CustomException(e, sys)

#Defining the custom class to input data through form
class CustomData: 
    def __init__(self,
                 cap_shape:str,
                 cap_surface:str,
                 cap_color:str,
                 bruises:str,
                 odor:str,
                 gill_attachment:str,
                 gill_spacing:str,
                 gill_size:str,
                 gill_color:str,
                 stalk_shape:str,
                 stalk_root:str,
                 stalk_surface_above_ring:str,
                 stalk_surface_below_ring:str,
                 stalk_color_above_ring:str,
                 stalk_color_below_ring:str,
                 veil_type:str,
                 veil_color:str,
                 ring_number:str,
                 ring_type:str,
                 spore_print_color:str,
                 population:str,
                 habitat:str):
        
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat #initializing every feature with self     
        logging.info("All features self initialized")   
    
    #defining function to store obtain data in the form of the dataframe
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "cap-shape" : [self.cap_shape],
                "cap-surface" : [self.cap_surface],
                "cap-color" : [self.cap_color],
                "bruises" : [self.bruises],
                "odor" : [self.odor],
                "gill-attachment" : [self.gill_attachment],
                "gill-spacing" : [self.gill_spacing],
                "gill-size" : [self.gill_size],
                "gill-color" : [self.gill_color],
                "stalk-shape" : [self.stalk_shape],
                "stalk-root" : [self.stalk_root],
                "stalk-surface-above-ring" : [self.stalk_surface_above_ring],
                "stalk-surface-below-ring" : [self.stalk_surface_below_ring],
                "stalk-color-above-ring" : [self.stalk_color_above_ring],
                "stalk-color-below-ring" : [self.stalk_color_below_ring],
                "veil-type" : [self.veil_type],
                "veil-color" : [self.veil_color],
                "ring-number" : [self.ring_number],
                "ring-type" : [self.ring_type],
                "spore-print-color" : [self.spore_print_color],
                "population" : [self.population],
                "habitat" : [self.habitat]  
            }     #storing obtain data to it relative coloumn
            
            #to input data into dataframe
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe is gathered as dataframe")
            return df
        
        #store the exception if occours          
        except CustomException as e:
            logging.info("Exception occuredd in prediction pipeline")
            raise CustomException(e, sys)
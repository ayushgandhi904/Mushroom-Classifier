import os,sys #for system path
from src.logger import logging #to log file
from src.exception import CustomException #to raise error as execption
import pandas as pd #handiling dataframe
from sklearn.model_selection import train_test_split #for splitting the data 
from dataclasses import dataclass #directly initializing the class

#data class for Data Ingestion --> For creating train & test data
@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join("artifacts", "train.csv") #path for joining the train data
    test_data_path:str = os.path.join("artifacts", "test.csv") #path for joininig the test data
    raw_data_path:str = os.path.join("artifacts", "raw.csv") #path for joining the raw data
    logging.info("Dataclass of Data Ingestion able to run sucessfully")
    
#Class for Data Ingestion
class DataIngestion:
    def __init__(self): #initializing with Dataingestion config
        self.ingestion_config = DataIngestionconfig()
        logging.info("Data Ingestion initialized")
        
    #Defining function for reading dataframe & splited into train & test
    def initiate_data_ingestion(self):
        logging.info("Data ingestion method started")
        
        try:
            df = pd.read_csv(os.path.join("notebooks/data", "mushrooms.csv")) #joining data path
            df["class"] = df["class"].apply(lambda x: {"p" : 1, "e": 0}[x]) #applying binary classifier to target variable
            logging.info("Dataset loaded by Dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True) #to create raw data in dirs, if doesn't exist
            df.to_csv(self.ingestion_config.raw_data_path, index = False) #Store the raw data in dir in csv form
            train_set, test_set = train_test_split(df, test_size = 0.3, random_state = 10) #split raw data into train & test 
            logging.info("Train-Test initialized from raw data")
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = "True") #store the train data in the directory
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True) #store the test data in the directory
            logging.info("Data ingestion step completed with data splitted in 70-30 ratio")
            
            return(
                self.ingestion_config.train_data_path, #returns train data
                self.ingestion_config.test_data_path #returns test data
            )
        
        #store the exception if occours    
        except Exception as e:
            logging.info("Exception occur at Data Ingestion step")
            raise CustomException(e, sys)
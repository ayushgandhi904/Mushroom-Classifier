from src.components.data_ingestion import DataIngestion #Data Ingestion module
from src.components.data_transformation import DataTransformation #Data Transformation module
from src.components.model_trainer import ModelTrainer #Model evaluation module

#connecting the pipeline with the all the stages

if __name__ == "__main__": #to run when initalized
    
    #1. Data Ingestion 
    obj = DataIngestion() 
    train_data_path, test_data_path = obj.initiate_data_ingestion() #return train & test data
    
    #2. Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path) #return transform data applied on train & test
    
    #3. Model Evaluation
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr) #return the model with best accuracy score to use it for the prediction pipeline
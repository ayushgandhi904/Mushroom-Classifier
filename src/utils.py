import os, sys, pickle #for system & saving pickele file
from src.logger import logging #for log file
from src.exception import CustomException #to raise custom exception if occurs
from sklearn.metrics import accuracy_score #accuracy score for the train data

#function for saving the object 
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) #path name      
        os.makedirs(dir_path, exist_ok= True) #path dir
        logging.info(f"{obj} able to load path")
        
        with open(file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj) #dump pickle file
        logging.info("File dump as pickle file")
    
    #store exception if occurs    
    except Exception as e:
        logging.info("File not able to dump as pickle")
        raise CustomException(e, sys)
    
 #defining function for evaluating accuracy of predicted & test data   
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {} #Creating Dict report
        logging.info("Creating model report")
        for i in range(len(models)):
            model = list(models.values())[i] #Listed all models
            logging.info("Model Listd in Dictionary")
            
            #Model Training
            model.fit(X_train, y_train)
            logging.info(f"{model} trained")
            
            #Predicting value
            y_test_pred = model.predict(X_test)
            logging.info(f"{model} predicted")
            
            #getting accuracy score
            test_model_score = accuracy_score(y_test, y_test_pred)
            logging.info(f"{model} accuracy score generated")
            
            report[list(models.keys())[i]] = test_model_score
            logging.info("Report generated")
            
        return report
    
    #storing excpetion if occurs
    except Exception as e:
        logging.info("Exception as model training step")
        raise CustomException(e, sys)

#function for loading the saved file
def load_object(file_path):
    
    try:        
        with open(file_path, "rb") as file_obj: #to open file
            logging.info("Pickle file loaded")
            return pickle.load(file_obj)
        
    #storing excpetion if occurs     
    except Exception as e:
        logging.info("Exception in utils load object")
        raise CustomException(e, sys)    
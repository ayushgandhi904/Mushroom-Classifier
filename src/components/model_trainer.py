import os, sys #for system
from src.logger import logging #for logging file
from src.exception import CustomException #for raise exception
from src.utils import save_object, evaluate_model #save object & evaluate model score
from dataclasses import dataclass #intializing the class
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV #LRegressor Model
from sklearn.tree import DecisionTreeClassifier #LRegressor Model
from sklearn.svm import SVC #LRegressor Model
from sklearn.neighbors import KNeighborsClassifier #LRegressor Model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #LRegressor Model

#Data class for Model Evaluation --> For storing the best model for predicting the data
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl") #storing the model file
    logging.info("Dataclass of Model Trainer able to run sucessfully")
    
#Class for Model Trainer    
class ModelTrainer:
    def __init__(self): #Initializing with Model Evaluation config
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("Model Trainer initialized")
        
    #Defining function on train model on train & test array    
    def initiate_model_training(self, train_array, test_array):
        try:
            #Separating Train & test array
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Data splitted into train & test array with dependent & independent array")
            
            #Creating Dictonary for models
            models = {
                "LogisticRegression" : LogisticRegression(),
                "LogisticRegressionCV" : LogisticRegressionCV(),
                "KNN" : KNeighborsClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "SVC" : SVC(),
                "RandomForest" : RandomForestClassifier(),
                "GradientBoosting" : GradientBoostingClassifier()
                
            }
            logging.info("Different models initalized")
           
            #appliying evaluation function on each model 
            model_report:dict=evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report) #to prin on code
            print("\n**********")
            logging.info(f"Model Report : {model_report}")
            
            #to get best score & model
            best_model_score = max(sorted(model_report.values()))
            logging.info("Model score sorted")
            
            #finding best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info("Best model name has been found")
            
            best_model = models[best_model_name]
            print(f"Best Model is {best_model_name} with accuracy : {best_model_score}")
            print("\n*****")
            logging.info(f"Best Model is {best_model_name} with accuracy : {best_model_score}")
            
            #to save the model file for further use
            save_object(   
                file_path = self.model_trainer_config.trained_model_file_path, #storing file path
                obj = best_model
            )
            logging.info("Best model saved as pkl file")
            
        #store the exception if occours  
        except Exception as e:
            logging.info("Exception occur at Model Training")
            raise CustomException(e, sys)
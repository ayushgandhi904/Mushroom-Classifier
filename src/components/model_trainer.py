import os, sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        
        try:
            logging.info("Splitting dependent & Independent data from train & test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "LogisticRegression" : LogisticRegression(),
                "LogisticRegressionCV" : LogisticRegressionCV(),
                "KNN" : KNeighborsClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "SVC" : SVC(),
                "RandomForest" : RandomForestClassifier(),
                "GradientBoosting" : GradientBoostingClassifier()
                
            }
            
            model_report:dict=evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            
            print("\n**********")
            logging.info(f"Model Report : {model_report}")
            
            #to get best score & model
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            print(f"Best Model is {best_model_name} with accuracy : {best_model_score}")
            print("\n*****")
            
            logging.info(f"Best Model is {best_model_name} with accuracy : {best_model_score}")
            
            save_object(
                
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
        except Exception as e:
            logging.info("Exception occur at Model Training")
            raise CustomException(e, sys)
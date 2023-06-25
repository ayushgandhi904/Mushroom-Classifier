import logging #for logging the file
import os #for system path
from datetime import datetime #datetime format

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #format for storing log
logs_path = os.path.join(os.getcwd(),'logs',LOG_FILE) #log path directory
os.makedirs(logs_path,exist_ok=True) #to creating new directory for log

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE) #join the log path

#level of logging --> INFO
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

#initializing the log
if __name__=="__main__": 
    logging.info("Logging has started!")
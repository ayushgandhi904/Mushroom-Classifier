import sys #for system

#defining error message
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() #3rd storage error is important, to store in the variable exc_tb
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]" #error message
    return error_message

#class for customexception when error occurs
class CustomException(Exception):
    
    def __init__(self, error_message, error_detail:sys): #initialization of error message
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self): #returning self message
        return self.error_message  
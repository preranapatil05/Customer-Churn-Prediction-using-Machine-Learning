import sys


# Creating function for error message detail
def error_message_detail(error, error_detail: sys):
    '''
    This function will return the error message with the file name and line number
    '''
    _, _, exc_tb = error_detail.exc_info()  # Getting information about the current exception

    file_name = exc_tb.tb_frame.f_code.co_filename  # extracting the file name where the error occured
    line_no = exc_tb.tb_lineno  # extracting the line number where the error occured

    error_message = "Error occured in python script name {0} line number [{1}] error message [{2}]".format(
        file_name, line_no,
        str(error))  # create the error message with file name, line number, and the actual error message

    return error_message


class CustomException(Exception):  # CustomException is a class that inherits from the built-in Exception class

    # This class is used to raise custom exception

    def __init__(self, error_message, error_detail: sys):
        '''
        This function will initialize the error message
        '''
        super().__init__(
            error_message)  # this fun calls the constructor of parent class Exception class and passes error message

        # Stores the error message along with file name and line number which is in error detail
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    # String representation of the CustomException object
    def __str__(self):
        '''
        This function will return the error message
        '''
        return self.error_message

import sys
from network_security.logging.logger import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error message including file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown File"
        line_number = "Unknown Line"

    return (
        f"Error occurred in script [{file_name}] at line [{line_number}]: {str(error)}"
    )


class NetworkSecurityException(Exception):
    def __init__(self, error_msg: Exception, error_detail: sys):
        """
        Custom Exception class for better debugging and logging.
        """
        self.error_msg = error_message_detail(error_msg, error_detail)

        # Log the error immediately
        logging.error(self.error_msg)

        # Optional: store original exception
        self.original_exception = error_msg

        super().__init__(self.error_msg)

    def __str__(self):
        return self.error_msg
    

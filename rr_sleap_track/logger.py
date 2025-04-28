import logging
import os

def setup_logger(name, log_out_folder):
    """
    Set up a logger with a specified name and log output folder.
    
    Parameters:
    name (str): The name of the logger.
    log_out_folder (str): The folder where log files will be stored.

    Returns:
    logging.Logger: Configured logger instance.
    """
    if not os.path.exists(log_out_folder):
        os.makedirs(log_out_folder)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create file handler to log to a file
    log_file = os.path.join(log_out_folder, f'{name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler to log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and set it for the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

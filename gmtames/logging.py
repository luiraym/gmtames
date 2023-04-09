''' 
gmtames
logging.py

Raymond Lui
26-July-2022
'''




from datetime import datetime
import pathlib




# DateTime|LoggingLevel|ModuleName.FuncName|PrimaryMsg(:|SecondaryMsg|etc.)
MESSAGE_FORMAT = '%(asctime)s|%(levelname)s|%(module)s.%(funcName)s|%(message)s'

DATETIME_LOG_EXTENSION = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')  + '.log'




def generateLoggingConfig(path_to_output):
    path_to_logs = path_to_output / 'logs/'
    path_to_logs.mkdir(exist_ok=True)

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'Formatter': {
                'format': MESSAGE_FORMAT,
            },
        },
        'handlers': {
            'FileHandler': {
                'class': 'logging.FileHandler',
                'formatter': 'Formatter',
                'filename': path_to_logs / ('gmtames' + DATETIME_LOG_EXTENSION),
                'mode': 'w',
                'delay': True,
            },
        },
        'loggers': {
            'gmtames': {
                'handlers': ['FileHandler'],
                'level': 'DEBUG',
            },
        },
    }

    return logging_config

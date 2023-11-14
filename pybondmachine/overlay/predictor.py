from pynq import Overlay
import numpy as np
import os
from .axisthandler import AxiStreamHandler

'''
model_specs = {
    "data_type": "float16",
    "register_size": "16",
    "batch_size": 16,
    "flavor": "axist",
    "n_input": 4,
    "n_output": 2
}
'''

def handle_config_error(func):
    def wrapper(*args, **kwargs):
        success, message = func(*args, **kwargs)
        if not success:
            raise ValueError(message)
        return success, message
    return wrapper

class Predictor():

    def __init__(self, firmware_name, firmware_path, model_specs):
        self.firmware_name = firmware_name
        self.firmware_path = firmware_path
        self.full_path = firmware_path+firmware_name
        self.model_specs = model_specs
        self.overlay = None
        self.X_test = None
        self.y_test = None

    def __load_overlay(self):
        try:
            self.overlay = Overlay(self.full_path)
        except Exception as e:
            return False, "Error loading the overlay. Error: "+str(e)
        
        return True, "Overlay loaded successfully"
    
    def __load_data(self, dataset_X, dataset_y):
        try:
            self.X_test = np.load(dataset_X)
            self.y_test = np.load(dataset_y)
        except Exception as e:
            return False, "Error loading the data. Error: "+str(e)
        
        return True, "Data loaded successfully"
    
    def __predict_axist(self):
        axisthandler = AxiStreamHandler(self.overlay, self.model_specs, self.X_test, self.y_test)
        return axisthandler.predict()

    def __predict(self):
        try:
            if (self.model_specs["flavor"] == "axist"):
                return True, self.__predict_axist()
            elif (self.model_specs["flavor"] == "aximm"):
                return False, "AXI-MM flavor not supported yet"
                # return self.__predict_aximm()
            else:
                return False, "Flavor not supported yet"
        except Exception as e:
            return False, "Error predicting the data. Error: "+str(e)
        
    @handle_config_error
    def load_overlay(self):
        return self.__load_overlay()
    
    @handle_config_error
    def load_data(self, dataset_X, dataset_y):
        return self.__load_data(dataset_X, dataset_y)
    
    @handle_config_error
    def predict(self):
        return self.__predict()


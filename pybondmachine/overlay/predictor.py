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
        self.handler = None

    def __load_overlay(self):
        try:
            self.overlay = Overlay(self.full_path)
            if (self.model_specs["flavor"] == "axist"):
                self.handler = AxiStreamHandler(self.overlay, self.model_specs)
        except Exception as e:
            return False, "Error loading the overlay. Error: "+str(e)
        
        return True, "Overlay loaded successfully"
    
    def __prepare_data(self, X_test, y_test):
        try:
            self.handler.X_test = X_test
            self.handler.y_test = y_test
            self.handler.prepare_data()
        except Exception as e:
            return False, "Error preparing the data. Error: "+str(e)
        
        return True, "Data prepared successfully"
    
    def __load_data(self, dataset_X, dataset_y):
        try:
            self.X_test = np.load(dataset_X)
            self.y_test = np.load(dataset_y)
            if self.handler is not None:
                self.handler.X_test = self.X_test
                self.handler.y_test = self.y_test
        except Exception as e:
            return False, "Error loading the data. Error: "+str(e)
        
        return True, "Data loaded successfully"
    
    def __predict_axist(self, debug=False):
        return self.handler.predict(debug)

    def __predict(self, debug=False):
        try:
            if (self.model_specs["flavor"] == "axist"):
                return True, self.__predict_axist(debug)
            elif (self.model_specs["flavor"] == "aximm"):
                return False, "AXI-MM flavor not supported yet"
                # return self.__predict_aximm()
            else:
                return False, "Flavor not supported yet"
        except Exception as e:
            return False, "Error predicting the data. Error: "+str(e)
        
    def __release(self):
        try:
            self.handler.release()
        except Exception as e:
            return False, "Error predicting the data. Error: "+str(e)

        return True, "Data released successfully"
    
    @handle_config_error
    def load_overlay(self):
        return self.__load_overlay()
    
    @handle_config_error
    def load_data(self, dataset_X, dataset_y):
        return self.__load_data(dataset_X, dataset_y)
    
    @handle_config_error
    def prepare_data(self, X_test, y_test):
        return self.__prepare_data(X_test, y_test)
    
    @handle_config_error
    def predict(self, debug=False):
        return self.__predict(debug)
    
    @handle_config_error
    def release(self):
        return self.__release()



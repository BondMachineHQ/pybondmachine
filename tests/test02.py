from pybondmachine.overlay.predictor import Predictor
import os
import numpy as np 

model_specs = {
    "data_type": "float32",
    "register_size": "32",
    "batch_size": 16,
    "flavor": "axist",
    "n_input": 4,
    "n_output": 2,
    "board": "zedboard",
}

firmware_name = "firmware.bit"
firmware_path = os.getcwd()

X_test = np.load(os.getcwd()+"/X_test.npy")
y_test = np.load(os.getcwd()+"/y_test.npy")

predictor = Predictor("firmware.bit", firmware_path, model_specs)
#predictor.load_data(os.getcwd()+"/X_test.npy", os.getcwd()+"/y_test.npy")
predictor.load_overlay()
predictor.prepare_data(X_test, y_test)
predictions = predictor.predict()
predictor.release()
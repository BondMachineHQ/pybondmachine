from pybondmachine.overlay.predictor import Predictor
import os

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

predictor = Predictor("firmware.bit", firmware_path, model_specs)
predictor.load_overlay()
predictor.load_data(os.getcwd()+"/X_test.npy", os.getcwd()+"/y_test.npy")
predictions = predictor.predict()

print(predictions)

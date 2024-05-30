import os
from pybondmachine.prjmanager.prjhandler import BMProjectHandler
from pybondmachine.converters.tensorflow2bm import mlp_tf2bm

#### THE FOLLOWING CODE IS A SAMPLE TO BUILD A FIRMWARE USING THE BONDMACHINE FRAMEWORK FROM PYTHON ####
#### BEFORE DOING THIS, YOU MUST TRAIN A NETWORK USING TENSORFLOW ####
#### IN THIS CASE, WE'LL USE A PRE-TRAINED MODEL ####

import tensorflow as tf
model = tf.keras.models.load_model(os.getcwd()+"/tests/model.h5")

output_file = "modelBM.json"
output_path = os.getcwd()+"/tests/"

# dump the json input file for neuralbond, the BM module that will be used to build the firmware
mlp_tf2bm(model, output_file=output_file, output_path=output_path)

prjHandler = BMProjectHandler("sample_project", "neuralnetwork", "/mlnotebook/projects_tests")

target_board = 'zedboard' # the target board for the firmware
prjHandler.check_dependencies() # check if the dependencies are installed
prjHandler.create_project(from_template=True, target_board=target_board) # create the project

config = {
    "data_type": "float16",
    "register_size": "16",
    "source_neuralbond": output_path+output_file,
    "flavor": "axist",
    "board": target_board
}

prjHandler.setup_project(config)
prjHandler.build_firmware()
<p align="center">
    <img src="images/pybondmachinelogo1.png" width="200" style="display:block; margin:auto;" />
</p>


# pybondmachine
pybondmachine is the Python library designed to streamline the development of FPGA accelerators through the use of the BondMachine framework.

With this library you can:
* (phase 1) Build a firmware starting from python code. You can use the BondMachine framework to create a neural network accelerator, or you can use the library to create a custom accelerator.
* (phase 2) Interact with the firmware to perform low-latency inference or to use the custom accelerator.


## Phase 1

### Prerequisites
* Python 3.6 or higher
* pip
* Vivado 2019.1 or higher (if you want to build the firmware)
* Tensorflow (to train or load a model)
* BondMachine framework (download it from [here](http://bondmachine.fisica.unipg.it/))

### Install
pip3 install pybondmachine

### Usage
*imports*
<pre>
from pybondmachine.prjmanager.prjhandler import BMProjectHandler
from pybondmachine.converters.tensorflow2bm import mlp_tf2bm
</pre>

*Load your neural network model (or train it from scratch)*
<pre>
import tensorflow as tf
model = tf.keras.models.load_model(os.getcwd()+"/tests/model.h5")
</pre>

*Convert your neural network model for BondMachine*
<pre>
output_file = "modelBM.json"
output_path = os.getcwd()+"/tests/"

# dump the json input file for neuralbond, the BM module that will be used to build the firmware
mlp_tf2bm(model, output_file=output_file, output_path=output_path)
</pre>

*Create and initialize a BM project with the params you prefer*
<pre>
prjHandler = BMProjectHandler("sample_project", "neuralnetwork", "projects_tests")

prjHandler.check_dependencies()
prjHandler.create_project()

config = {
    "data_type": "float16",
    "register_size": "16",
    "source_neuralbond": output_path+output_file,
    "flavor": "axist",
    "board": "zedboard"
}

prjHandler.setup_project(config)
</pre>
*Build the firmware*
<pre>
prjHandler.build_firmware()
</pre>

## Phase 2

### Prerequisites
* Python 3.6 or higher
* pip
* Pynq (if you want to use the custom accelerator)
* FPGA device

*Load the predictor*
<pre>
from pybondmachine.overlay.predictor import Predictor
</pre>

*Set the model specs*
<pre>
model_specs = {
    "data_type": "fps16f6",
    "register_size": 16,
    "batch_size": 128,
    "flavor": "axist",
    "n_input": 4,
    "n_output": 2,
    "benchcore": True,
    "board": "zedboard"
}
</pre>

*Specify firmware name and firmware path*
<pre>
firmware_name = "firmware.bit"
firmware_path = "proj_zedboard_axist_fp16_6_expanded_01/"
</pre>

*Initialize the predictor*
<pre>
predictor = Predictor("firmware.bit", firmware_path, model_specs)
</pre>
*Load the data to be processed*
<pre>
predictor.load_data("proj_zedboard_axist_fp16_6_expanded_01/banknote-authentication_X_test.npy", 
                    "proj_zedboard_axist_fp16_6_expanded_01/banknote-authentication_y_test.npy")
</pre>
*Load the overlay i.e. program the FPGA*
*Remember that it is necessary that you call predictor.load_data before loading the overlay*
<pre>
predictor.load_overlay()
</pre>
*Perform inference*
<pre>
status, predictions = predictor.predict()
</pre>


## Phase 1 details under the hood
This python package is basically a wrapper of the BondMachine helper tool. It allows you to create a project, to build the firmware and to convert a neural network model to a json file that can be used as input for the BondMachine framework. Or, if you prefer, you can use the library to create a custom accelerator.
Under the hood, **bmhelper** create the project, modify all the parameters of the configuration files inside the project and it checks the dependencies. 
Indeed, you can use bmhelper from CLI if you have installed it.
So, to use the python library bmhelper is necessary and you can get it from here: [bmhelper](http://bondmachine.fisica.unipg.it/ug/) under the section "Installation".


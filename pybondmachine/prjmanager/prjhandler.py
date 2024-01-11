import os
from .neuralnetwork import NeuranNetworkHandler
import subprocess

def handle_config_error(func):
    def wrapper(*args, **kwargs):
        success, message = func(*args, **kwargs)
        if not success:
            raise ValueError(message)
        return success, message
    return wrapper

class BMProjectHandler():

    def __init__(self, project_name, project_type, output_folder=""):
        self.project_name = project_name
        self.project_type = project_type
        self.project_path = os.getcwd() + '/' + output_folder + '/'
        self.project_full_path = self.project_path + self.project_name + '/'
        
    def __check_dependencies(self):
        current_os = os.name
        if (current_os == "posix"):
            try:
                # get the output command of the bmhelper version
                bmhelper_cmd = subprocess.Popen("bmhelper version", shell=True, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
                out, err = bmhelper_cmd.communicate()
                out_string = out.decode("utf-8")

                if err is not None:
                    return False, "Error checking dependencies. Error: "+err.decode("utf-8")

                if ("not found" in out_string):
                    return False, "Dependencies not satisfied. Please install the BondMachine framework (Get bmhelper tool https://github.com/BondMachineHQ/bmhelper to have more details)"
                
            except Exception as e:
                return False, "Error checking dependencies. Error: "+str(e)
        else:
            return False, "Operating system not supported"
        
        # call the command "bmhelper doctor" and return the output to the user
        try:
            bmhelper_cmd = subprocess.Popen("bmhelper doctor", shell=True, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
            out, err = bmhelper_cmd.communicate()
            out_string = out.decode("utf-8")

            if err is not None:
                return False, "Error checking dependencies. Error: "+err.decode("utf-8")
        except Exception as e:
            return False, "Error checking dependencies. Error: "+str(e)
        
        return True, "Dependencies checked successfully"

    
    # The following method is responsible for creating a new project. Here's a step-by-step breakdown of what it does:
    # 1. It checks if the project type is "neuralnetwork". If it's not, it returns (for the momement) an error message saying that the project type is not supported yet.
    # 2. If the project type is "neuralnetwork", it checks if the output folder exists. If it doesn't, it creates the folder.
    # 3. It then changes the current working directory to the project path.
    # 4. If the `from_template` parameter is `True`, it calls the `bmhelper` command-line tool to create a new project from a template. If `from_template` is `False`, it creates a new project without using a template.
    # 5. If there's an error while executing the command, it catches the exception and returns an error message.
    # 6. Finally, it checks if the project directory has been created. If it has, it returns a success message. If it hasn't, it returns an error message.
    
    def __create_project(self, from_template=True, target_board=None):

        # if the folder already exists, return an error message
        if (os.path.isdir(self.project_full_path)):
            return False, "Project already exists"

        # based on the project type, create the project starting from template or from scratch
        if (self.project_type == "neuralnetwork"):
            
            # check if output folder length is > 0 and if it is greater than 0 check if it exists, if not create it
            if (len(self.project_path) > 0):
                if (not os.path.isdir(self.project_path)):
                    os.makedirs(self.project_path)
            
            # handle the eventual error executing the command line
            try:
                # call the bmhelper cli tool in order to create the project using the following command line: bmhelper create --project_name project_test --example example_name
                # but be sure you are in the project path folder
                os.chdir(self.project_path)
                if (from_template):
                    if (target_board == 'zedboard'):
                        os.system("bmhelper create --project_name " + self.project_name + " --example proj_zedboard_ml_creation") # to change with the correct template example
                    elif(target_board == 'alveou50'):
                        os.system("bmhelper create --project_name " + self.project_name + " --example proj_alveou50_ml_accelerator")
                    else:
                        os.system("bmhelper create --project_name " + self.project_name + " --example proj_zedboard_ml_creation") 
                else:
                    os.system("bmhelper create --project_name " + self.project_name) # to change with the correct template example
                    
            except Exception as e:
                return False, "Error creating the project. Error: "+str(e)
        else:
            # print an error message for now and exit
            return False, "Project type not supported yet"
        
        # check if the project has been created
        if (os.path.isdir(self.project_full_path)):
            return True, "Project created successfully"
        else:
            return False, "Error creating the project"
    

    def __apply_configurations(self):
        try:
            os.chdir(self.project_full_path)
            os.system("bmhelper apply")
        except Exception as e:
                return False, "Error creating the project. Error: "+str(e)
        
        return True, "Project successfully configured, ready to build the firmware"
    
    def setup_project(self, project_specs):
        # call the correct straregy to setup the project based on the project type
        if (self.project_type == "neuralnetwork"):
            try:
                nn_handler = NeuranNetworkHandler(self.project_full_path, project_specs)
                nn_handler.setup_project()
            except Exception as e:
                return False, "Error setting up the project. Error: "+str(e)
        else:
            # print an error message for now and exit
            return False, "Project type not supported yet"
    
        return True, "Project setup successfully"
    
    
    def __build_firmware(self, oncloud=False):
        
        self.__apply_configurations()
        if oncloud == False:
            try:
                cmd_output = subprocess.check_output("bmhelper doctor", shell=True, text=True)
                if "BondMachine tool not found" in cmd_output:
                    return False, "BondMachine tool not found. Please install the BondMachine framework"

                board_in_use = ''
                with open(self.project_full_path+"/local.mk", 'r') as f:
                    for line in f:
                        if "BOARD" in line:
                            board_in_use = line[line.index('=')+1:len(line)]
                            break
                
                if (board_in_use.startswith("alveo")):
                    os.system("make xclbin")
                    # wip: create a json file that describes the accelerator properties
                    # an example is under tests folder called "accelerator.json"
                else:
                    os.system("make design_bitstream")
            
            except Exception as e:
                return False, "Unable to build the firmware on local system. Error: "+str(e)    
        else:
            # TODO: add the command to build the firmware on the cloud
            print("To do")

        return True, "Firmware built successfully"

    @handle_config_error
    def check_dependencies(self):
        return self.__check_dependencies()
    
    @handle_config_error
    def create_project(self, from_template=True, target_board=None):
        return self.__create_project(from_template, target_board)
    
    @handle_config_error
    def build_firmware(self, oncloud=False):
        return self.__build_firmware(oncloud)
'''
project specs is a dictionary containing the configuration parameters for the project
{
    "data_type": "float16",
    "register_size": "16",
    "source_neuralbond": "modelBM.json",
    "flavor": "axist"
    "board": "zedboard"
}
'''
import os

class NeuranNetworkHandler():
    def __init__(self, project_path, project_specs):
        self.project_path = project_path
        self.project_specs = project_specs
        self.configurations_file = [
            "local.mk",
            "deploy.mk",
            "bmapi.mk"
        ]
    
    # create a private method to modify local.mk configuration file
    def __modify_local_mk(self):
        lines_to_save = []
        with open(self.project_path+"local.mk", 'r') as f:
            for line in f:
                if "NEURALBOND_ARGS=" in line:
                    new_line = "NEURALBOND_ARGS=-config-file neuralbondconfig.json -operating-mode fragment -io-mode sync -data-type "+self.project_specs["data_type"]+" -register-size "+self.project_specs["register_size"]+"\n"
                    lines_to_save.append(new_line)
                elif "SOURCE_NEURALBOND" in line:
                    new_line = "SOURCE_NEURALBOND=modelBM.json\n"
                    lines_to_save.append(new_line)
                elif "BOARD" in line:
                    new_line = "BOARD="+self.project_specs["board"]+"\n"
                    lines_to_save.append(new_line)
                else:
                    lines_to_save.append(line)
                    
        # save the file local.mk with the new lines
        with open(self.project_path+"local.mk", 'w') as f:
            for line in lines_to_save:
                f.write(line)
    
    def __modify_bmapi_mk(self):
        lines_to_save = []
        with open(self.project_path+"bmapi.mk", 'r') as f:
            for line in f:
                if "BMAPI_DATATYPE=" in line:
                    new_line = "BMAPI_DATATYPE="+self.project_specs["data_type"]+"\n"
                    lines_to_save.append(new_line)
                else:
                    lines_to_save.append(line)
                    
        # save the file local.mk with the new lines
        with open(self.project_path+"bmapi.mk", 'w') as f:
            for line in lines_to_save:
                f.write(line)
    
    def setup_project(self):

        # check if source neural network file exists
        if not os.path.exists(self.project_specs["source_neuralbond"]):
            raise Exception("Source neural network file not found")

        # copy the source neural network file to the project path
        os.system("cp "+self.project_specs["source_neuralbond"]+" "+self.project_path+"modelBM.json") 

        # set the project configurations file
        try:
            for file in self.configurations_file:
                if file == "local.mk":
                    self.__modify_local_mk()  
                elif file == "bmapi.mk":
                    self.__modify_bmapi_mk()  
                    
        except Exception as e:
            raise Exception("Error setting up the project. Error: "+str(e))
        
        return True, "Project setup successfully"
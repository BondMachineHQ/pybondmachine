import json
import os

def mlp_tf2bm(model, output_file="modelBM.json", output_path="models/"):
    neural_networks_params = []
    layers = model.layers
    weights = model.weights

    to_dump = {}

    weights = []
    nodes = []

    weights_value = []

    # save weigths
    for i in range(0 , len(layers)):

        layer_weights = layers[i].get_weights()

        for m in range(0, len(layer_weights)):
            for w in range(0, len(layer_weights[m])):
                try:
                    for v in range(0, len(layer_weights[m][w])):
                        
                        if float(layer_weights[m][w][v]) == 0:
                            continue
                        
                        weights_value.append(float(layer_weights[m][w][v]))

                        weight_info = {
                                "Layer": i+1,
                                "PosCurrLayer": v,
                                "PosPrevLayer": w,
                                "Value": float(layer_weights[m][w][v])
                            }
                        neural_networks_params.append(float(layer_weights[m][w][v]))
                        weights.append(weight_info)
                except:
                    continue

        if i == 0:
            for units in range(0, layers[i].units):
                weights_l0 = layers[i].get_weights()[0]

                for w in range(0, len(weights_l0)):
                    node_info = {
                        "Layer": 0,
                        "Pos": w,
                        "Type": "input",
                        "Bias": 0
                    }
                    neural_networks_params.append(0)
                    nodes.append(node_info)
                break
        

        for units in range(0, layers[i].units):
            if i == len(layers) - 1:
                bias = layers[i].get_weights()[1]
                node_info = {
                    "Layer": i+1,
                    "Pos": units,
                    "Type": "summation",
                    "Bias": bias.tolist()[units]
                }
                nodes.append(node_info)
                neural_networks_params.append(float(node_info["Bias"]))
                weights_value.append(node_info["Bias"])
            else:
                name = ""
                try:
                    name = layers[i].activation.__name__
                except Exception as e:
                    name = str(layers[i].activation)
                    
                bias = layers[i].get_weights()[1]
                node_info = {
                    "Layer": i+1,
                    "Pos": units,
                    "Type": name,
                    "Bias": bias.tolist()[units]
                }
                nodes.append(node_info)
                neural_networks_params.append(float(node_info["Bias"]))
                weights_value.append(node_info["Bias"])

        if i == len(layers) - 1:
            for l in range(0, len(layer_weights[0][0])):
                node_info = {
                    "Layer": i+2,
                    "Pos": l,
                    "Type": "softmax",
                    "Bias": 0
                }
                nodes.append(node_info)


            layer_weights = layers[i-2].get_weights()

            for k in range(0, layers[i].units):
                for l in range(0, layers[i].units):
                    weight_info = {
                                    "Layer": i+2,
                                    "PosCurrLayer": k,
                                    "PosPrevLayer": l,
                                    "Value": 1
                                }
                    neural_networks_params.append(float(1))
                    weights.append(weight_info)


        if i == len(layers) - 1:

            layer_weights = layers[i].get_weights()

            for l in range(0, len(layer_weights[0][0])):
                node_info = {
                    "Layer": i+3,
                    "Pos": l,
                    "Type": "output",
                    "Bias": 0
                }
                nodes.append(node_info)

                weight_info = {
                                "Layer": i+3,
                                "PosCurrLayer": l,
                                "PosPrevLayer": l,
                                "Value": 1
                            }
                weights.append(weight_info)

    to_dump["Nodes"] = nodes
    to_dump["Weights"] = weights

    # if output path does not exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # dump the json file inside the output path
    with open(output_path+output_file, 'w') as fp:
        json.dump(to_dump, fp)
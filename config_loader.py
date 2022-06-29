import json


def load_config(json_filepath):
    new_dict = dict()
    with open(json_filepath) as json_file:
        data = json.load(json_file)
        for key in data.keys():
            if key != "_wandb":
                new_dict[key] = data[key]["value"]
    return new_dict



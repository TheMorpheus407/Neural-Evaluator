import os
import json
from flask import abort


mydir = "Sites/"
attacks_file = "NeuralEvaluator/website_attacks.txt"
def create_set(url, raw_website, attacked_website, payload, successful):
    """creates a new training example"""
    with open(attacks_file, "r") as f:
        all_data = json.load(f)
        file_path = mydir + url.split("/")[-1] + ".raw"
        if not os.path.isfile(file_path):
            open(file_path, "w").write(raw_website)
        i = 0
        attack_path = mydir + url.split("/")[-1] + "_" + str(i) + ".raw"
        while os.path.isfile(attack_path):
            i = i + 1
            attack_path = mydir + url.split("/")[-1] + "_" + str(i) + ".raw"

        open(attack_path, "w").write(attacked_website)

        if successful:
            # POSITIVE-Data
            all_data.append({"file": file_path, "attacked_file": attack_path, "payload": payload, "target": "1-10", "method": "post"})
        else:
            # NEGATIVE-Data
            all_data.append({"file": file_path, "attacked_file": attack_path, "payload": payload, "target": "-", "method": "post"})
    f = open(attacks_file, "w")
    json.dump(all_data, f, indent=2)

def read_sets():
    """returns all current training sets"""
    with open(attacks_file, "r") as f:
        all_data = json.load(f)
    return {'items': all_data}

def delete_set(filename):
    """removes a training set - both the file and the entry in the json. CAN NOT BE UNDONE!"""
    try:
        with open(attacks_file, "r") as f:
            all_data = json.load(f)
        for i in all_data:
            if i["attacked_file"].split("/")[-1] == filename:
                if os.path.isfile(mydir+filename):
                    os.remove(mydir+filename)
                all_data.remove(i)
        f = open(attacks_file, "w")
        json.dump(all_data, f, indent=2)
    except Exception as e:
        print(e)
        abort(404)
    return "ok"

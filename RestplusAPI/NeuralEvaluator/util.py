import difflib
import torch
import string
import random
import unicodedata
from torch.autograd import Variable
import json
import numpy as np
import requests
import os

dirname = os.path.dirname(__file__)
targetstrings = os.path.join(dirname,json.load(open(os.path.join(dirname, 'config.json')))["trainexamples"])
website_attacks = os.path.join(dirname,json.load(open(os.path.join(dirname, 'config.json')))["website_attacks"])

def get_EOS_token():
    return "ð"

def get_letters():
    return string.punctuation + " " + get_EOS_token() + string.ascii_letters + string.digits

def get_letters_num():
    return len(get_letters())

def to_ascii(s):
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn'
        and char in get_letters()
    )

def lines(datei):
    f = open(os.path.join(dirname,datei), encoding='utf-8').read().split('\n')
    return [to_ascii(l) for l in f]

def char_to_index(char):
    return get_letters().find(char)

def char_to_tensor(char):
    ret = torch.zeros(1, get_letters_num()) #ret.size = (1, letters_num)
    ret[0][char_to_index(char)] = 1
    return ret

def example_to_tensor(example):
    ret = torch.zeros(1, len(example), get_letters_num())
    for i, char in enumerate(example):
        ret[0][i][char_to_index(char)] = 1
    return ret

def target_char_to_tensor(target):
    indizes = [get_letters().find(target)]
    return torch.LongTensor(indizes)

def batch_example_to_tensor(batch):
    out = []
    for i in batch:
        out.append(example_to_tensor(i))
    return torch.cat(out)

def tensor_to_char(tensor):
    try:
        idx = (np.where(tensor.cpu().numpy() == tensor.max()))[0][0]
        return get_letters()[idx]
    except:
        return get_EOS_token()

def tensor_to_string(tensor):
    output = ""
    if tensor.dim() == 3:
        for i in tensor[0]:
            output = output + tensor_to_char(i)
    else:
        for i in tensor:
            output = output + tensor_to_char(i)
    return output

def target_to_tensor(target):
    indizes = [get_letters().find(target[i]) for i in range(1,len(target))]
    indizes.append(get_letters_num() - 1)
    return torch.LongTensor(indizes)

def get_website_attacks(type='json'): #TODO generate automatically, Whitespaces, Kommata in Datei..
    if type == "json":
        with open(website_attacks, 'r') as f:
            js = json.load(f)
            random.shuffle(js)
            for element in js:
                payload = element["payload"]
                raw_target = element["target"]
                url = element["url"]
                resp = requests.get(url)
                headers = prepare_headers(resp.headers)
                text = resp.text.replace("\t", "").replace("\r\n", "").replace("\n", "")
                yield payload, (headers, text), (raw_target.split("-")[0], raw_target.split("-")[1]) #old version. there is no url anymore! Now with files. See differences

def get_website_attacks_differences(type='json'): #TODO generate automatically, Whitespaces, Kommata in Datei..
    if type == "json":
        with open(website_attacks, 'r') as f:
            js = json.load(f)
            random.shuffle(js)
            for element in js:
                payload = element["payload"]
                raw_target = element["target"].split("-")
                raw = str(open(element["file"]).read(200000)).replace("\t", "").replace("\r\n", "").replace("\n", "")
                attacked = str(open(element["attacked_file"]).read(200000)).replace("\t", "").replace("\r\n", "").replace("\n", "")
                output_list = get_string_difference(raw, attacked)
                if len(output_list) == 0:
                    output_list = [" "]
                yield payload, raw_target, end_diffs(padd_differences(output_list))


def padd_payload(payload):
    while len(payload) > 100:
        payload = payload[:100]
    while len(payload) < 100:
        payload += ' '
    return payload

def padd_differences(diffs):
    while len(diffs) > 100:
        diffs.pop()
    while len(diffs) < 100:
        diffs.append('+ ')
    return diffs

def end_diffs(diffs):
    diffs = [diffs[i][-1] for i in range(len(diffs))]
    return diffs

def get_string_difference(a, b):
    output_list = [li for li in list(difflib.ndiff(a,b)) if li[0] != ' ']
    return end_diffs(padd_differences(output_list))

def prepare_headers(headers):
    if "date" in headers:
        headers.pop("date")
    headers = ''.join([x + headers[x] for x in headers])
    return headers.replace("\t", "").replace("\r\n", "").replace("\n", "")

def generate_target_vuln(headers, text, target, target_is_index_of_total_site = False):
    target_tensor = torch.zeros(1,len(headers)+len(text), 1)
    if target[0] != "" and target[1] != "":
        start = int(target[0])
        end = int(target[1])
        if not target_is_index_of_total_site:
            start += len(headers)
            end += len(headers)
        for i in range(start, end):
            target_tensor[0][i][0] = 1
    return target_tensor

def generate_target_vuln_fullsite(site, target, target_is_index_of_total_site = False):
    target_tensor = torch.zeros(1, 1, 1)
    if target[0] != "" and target[1] != "":
        target_tensor = torch.ones(1, 1, 1)
    return target_tensor


def payloaddict_to_string(payload_dict, payload):
    to_inject = {}
    counter = 0
    for j in payload_dict:
        if "ZAP" in payload_dict[j] and counter < len(payload):
            s = payload_dict[j].replace("ZAP", payload[counter], 1)
            counter = counter + 1
            to_inject.update({j: s})
        else:
            to_inject.update({j: payload_dict[j]})
    return to_inject

lines_file = lines(targetstrings)
lines_copied = lines(targetstrings)
def get_random_example():
    return random.choice(lines_file)

def get_all_targets_copied():
    return lines_copied

def get_random_train():
    pw = get_random_example()
    input_tensor = Variable(example_to_tensor(pw))
    target_tensor = Variable(target_to_tensor(pw))
    return input_tensor, target_tensor

def get_all_targets():
    return lines_file

def get_html(path = 'top-1m.csv', type='csv'):
    if type == "csv":
        with open(path, 'r') as f:
            for line in f:
                url = line.split(',')[1][:-1] #cutting off \n TODO USE remove - last line?
                resp = requests.get("http://" + url)
                headers = prepare_headers(resp.headers)
                text = resp.text.replace("\r\n", "").replace("\n", "").replace("\t", "")
                yield headers, text
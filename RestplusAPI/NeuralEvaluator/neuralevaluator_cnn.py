import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import RestplusAPI.NeuralEvaluator.util as util
import os

is_cuda = False
FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if is_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if is_cuda else torch.ByteTensor
Tensor = FloatTensor

hidden_size = 512
num_layers = 4

model_file = os.path.join(os.path.dirname(__file__), "neural_evaluator_model.pt")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = hidden_size, dropout = 0.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        #self.lstm = nn.LSTM(util.get_letters_num(), hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.conv1 = nn.Conv2d(1,3,(5,5))
        self.conv2 = nn.Conv2d(3,6,(5,5))
        self.conv3 = nn.Conv2d(6,12,(5,5))
        self.conv4 = nn.Conv2d(12,24,(5,5))
        #self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(384,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, x):
        if x.size()[0] != 1 or x.size()[1] != 200 or x.size()[2] != 96:
            return torch.zeros(1,1)
        x = x.view(1,1,x.size()[1],x.size()[2]) #1,1,200,96
        x = nn.MaxPool2d(2)(self.conv1(x))
        x = self.dropout(F.relu(x)) #1,3,96,46
        x = nn.MaxPool2d(2)(self.conv2(x))
        x = self.dropout(F.relu(x)) #1,6,47,21
        x = nn.MaxPool2d(2)(self.conv3(x))
        x = self.dropout(F.relu(x)) #1,12,21,8
        x = nn.MaxPool2d(2)(self.conv4(x))#1,24,8,2
        x = x.view(1,-1)#1,384
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return F.sigmoid(x)

'''later needed for generation of payloads'''
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, input_size = hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #self.lstm = nn.LSTMCell(hidden_size, util.get_letters_num(), num_layers)
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):#, output_length):
        '''hx_tensor = torch.zeros(1, util.get_letters_num())
        cx_tensor = torch.zeros(1, util.get_letters_num())
        if is_cuda:
            hx_tensor = hx_tensor.cuda()
            cx_tensor = cx_tensor.cuda()
        hx = Variable(hx_tensor)
        cx = Variable(cx_tensor)
        output = []
        for i in range(output_length):
            hx,cx = self.lstm(encoded_input, (hx, cx))
            hx = self.sigmoid(hx)
            output.append(hx)
        return torch.cat(output)'''
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output

'''later needed for generation of payloads'''
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_path = None, train_model = True):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers)
        if is_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.path = model_path
        self.train_model = train_model

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)#, input.size()[1])
        return decoded_output

    def save_model(self):
        if self.train and self.path is not None:
            torch.save(self, self.filename)

    def train_net(self, lr=0.001):
        running_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        losses = []
        optimizer = optim.Adam(self.parameters(), lr)
        for i, item in enumerate(util.get_html()):
            start = time.time()
            input_tensor = util.example_to_tensor((item[0]+item[1])[:2])
            if is_cuda:
                input_tensor = input_tensor.cuda()
            input_var =  Variable(input_tensor)
            output = self(input_var)
            optimizer.zero_grad()
            #exit()
            loss = criterion(output, input_var)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%5d] loss: %.3f' % (i + 1, running_loss / 2000))
                losses.append(running_loss/2000)
                running_loss = 0.0
                self.save_model()
            print("Epoch took " + str(time.time() - start) + " to complete")
        return losses

'''
always queried - prints out where its vuln., if there is none, the website is safe
'''
class NeuralEvaluatorModel(nn.Module):
    def __init__(self, hidden_size = hidden_size, num_layers = num_layers, model_path = model_file, train_model = True, intermediate_size=16):
        super(NeuralEvaluatorModel, self).__init__()
        self.website_encoder = EncoderRNN(util.get_letters_num(), hidden_size, num_layers, output_size=intermediate_size)
        #self.payload_encoder = EncoderRNN(util.get_letters_num(), hidden_size, num_layers, output_size=intermediate_size)
        #self.linear = nn.Linear(in_features=intermediate_size, out_features=1)
        #self.sigmoid = nn.Sigmoid()
        if is_cuda:
            self.website_encoder = self.website_encoder.cuda()
            #self.payload_encoder = self.payload_encoder.cuda()
        self.path = model_path
        self.train_model = train_model

    def forward(self, website, payload):
        website = torch.cat((website, payload), 1)
        x = self.website_encoder(website) #Size: (1,length,8)
        #payload = self.payload_encoder(payload)
        #exit()
        #payload = payload[len(payload)-1].view(1,1,-1) #Size: (1,1,8)
        #x_payload = torch.cat([payload for _ in range(website.size()[1])], 1) #Size: (1,length,8)
        #x = torch.cat((website, x_payload), 2) #Size: (1,length,16)
        return x #self.sigmoid(self.linear(x)) #Size: (1,length,1)

    def save_model(self):
        if self.train and self.path is not None:
            torch.save(self, self.path)

    def train_net(self, lr=0.001):
        running_loss = 0.0
        criterion = nn.MSELoss()
        losses = []
        optimizer = optim.Adam(self.parameters(), lr)
        for i, (payload, target, difference) in enumerate(util.get_website_attacks_differences()): #TODO FULL payload (DICT!!), website: (header, text), target:(start idx von vuln., länge)
            if payload is None or difference is None or target is None:
                continue
            start = time.time()
            payload = util.padd_payload(str(payload))
            website_tensor = util.example_to_tensor(difference)
            payload_tensor = util.example_to_tensor(payload)
            target = util.generate_target_vuln_fullsite(difference, target)
            if is_cuda:
                website_tensor = website_tensor.cuda()
                payload_tensor = payload_tensor.cuda()
                target = target.cuda()
            website_var = Variable(website_tensor)
            payload_var = Variable(payload_tensor)
            output = self(website_var, payload_var) #Size: (1,length,1)
            optimizer.zero_grad()
            loss = criterion(output, Variable(target))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            print('[%5d] loss: %.3f' % (i + 1, loss.data[0]))
            losses.append(loss.data[0])
        self.save_model()
        #print("Epoch took " + str(time.time() - start) + "s to complete")
        return losses

class NeuralEvaluator():
    def __init__(self, evaluator = None):
        if evaluator == None:
            self.evaluator = NeuralEvaluatorModel()
        else:
            self.evaluator = evaluator
        if is_cuda:
            self.evaluator = self.evaluator.cuda()


    #def register_callback_function(self, func):
        #self.callbacks.append(func)

    #def delete_callback_function(self, func):
        #self.callbacks.remove(func)

    def predict(self, raw_website, attacked_website, payload):
        diff = util.get_string_difference(raw_website, attacked_website)
        if len(diff) == 0:
            diff = [" "]
        payload = util.padd_payload(payload)
        website_tensor = util.example_to_tensor(diff)
        payload_tensor = util.example_to_tensor(payload)
        if is_cuda:
            website_tensor = website_tensor.cuda()
            payload_tensor = payload_tensor.cuda()
        website_var = Variable(website_tensor)
        payload_var = Variable(payload_tensor)
        output = self.evaluator(website_var, payload_var) #Size: (1,length,1)
        return output


def load_neuralevalmodel_from_file(model_path, hidden_size=hidden_size, num_layers = num_layers):
    aa = NeuralEvaluatorModel(hidden_size=hidden_size, num_layers = num_layers, model_path = model_path)
    if model_path is not None and os.path.isfile(model_path):
        aa = torch.load(model_path)
    return aa

def train(losses_path="losses_neural_evaluator.txt", model_path = model_file):
    epochs = 20
    aa = load_neuralevalmodel_from_file(model_path)
    if is_cuda:
        aa = aa.cuda()
    losses = []
    for i in range(epochs):
        start = time.time()
        print("###############################")
        print("########    EPOCH " + str(i) + "    ########")
        print("###############################")
        losses.append(aa.train_net())
        print("Epoch took " + str(time.time()-start) + " s to complete")
    f = open(losses_path, "w")
    json.dump(losses, f, indent=2)

def predict(raw_website, attacked_website,  payload):
    model = NeuralEvaluator(evaluator=load_neuralevalmodel_from_file(model_file))
    res = model.predict(raw_website, attacked_website, payload)
    if res.data[0][0] >= 0.5:
        output = True
    else:
        output = False
    #print(output)
    return output

if __name__ == "__main__":
    # Train the AA!
    train()
    exit()
    with open("website_attacks_validation.txt", 'r') as f:
        js = json.load(f)
    results = []
    for i in js:
        if "message" in i["payload"]:
            result = predict(open(i["file"]).read(), open(i["attacked_file"]).read(), payload=str(i["payload"]["message"])) == (len(i["target"].split("-")[0])>0)
            results.append([result, (len(i["target"].split("-")[0])>0)])
    correct = len([i for i in results if i[0]])
    true_pos = len([i for i in results if i[0] and i[1]])
    false_pos = len([i for i in results if not i[0] and not i[1]])
    true_neg = len([i for i in results if i[0] and not i[1]])
    false_neg = len([i for i in results if not i[0] and i[1]])

    print("Accuracy: " + str(correct/len(results) * 100) + " %")
    print("Sensitivity: " + str(true_pos / (true_pos + false_neg)))
    print("Specificity: " + str(true_neg / (true_neg + false_pos)))
    print("Precision: " + str(true_pos / (true_pos + false_pos)))
    print("False-Positive Rate: " + str(false_pos / (false_pos + true_neg)))
    exit()
    predict(open("To_Predict/xss.php.raw").read(), open("To_Predict/xss.php_0.raw").read(), payload="<SCRIPT>alert('')</SCRIPT>")
    predict(open("To_Predict/xss.php.raw").read(), open("To_Predict/xss.php_1.raw").read(), payload="foobar asdf hello this is goodscript")
    predict(open("To_Predict/xss.php.raw").read(), open("To_Predict/xss.php_2.raw").read(), payload="<script>injectsomefunction();</SCRIPT>")
    predict(open("To_Predict/xss.php.raw").read(), open("To_Predict/xss.php_3.raw").read(), payload="' or ''='")



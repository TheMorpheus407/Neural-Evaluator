# NEURAL EVALUATOR MICROSERVICE

This is a wrapping microservice API for using the neural network for evaluating attack vectors on a website.
It uses Python Flask for the API and PyTorch for the neural network.

This project is a cooperation of C&M of the Institute for Technology in Karlsruhe, Germany and IC-Consult. It was created as part of the master thesis of Cedric Mössner.
You can see more information in the thesis paper found here:https://drive.google.com/open?id=1Gqq4D6mqnqFaYFtwZKSPALJu1LcqcU8z 
If you still need more information, please feel free to contact me at kontakt@the-morpheus.de.

### USAGE with DOCKER
If not already done, install docker from [here](https://www.docker.com/get-started)

Use the Dockerfile to create a docker image:
```
docker build -t evaluatormicroservice .
docker run -p 1337:1337 evaluatormicroservice
```

### USAGE without DOCKER
Get [Python 3](https://www.python.org/) and [Pip](https://pip.pypa.io/en/stable/installing/).
Then do:
```sh
$ pip -r requirements.txt
```

On Linux then go ahead and install PyTorch for CPU:
```
$ pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl 
$ pip3 install torchvision
```
or follow the instructions at [PyTorch.org](https://pytorch.org/)

then you can run it with

```sh
$ python app.py
```

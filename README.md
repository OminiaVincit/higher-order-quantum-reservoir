# Higher-Order Quantum Reservoir Computing

This repository is the official implementation of Higher-Order Quantum Reservoir Computing. 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements
The code requires the following libraries:
- python 3.6-3.7
- tensorflow 1.11.0-1.15.0
- numpy, matplotlib, sklearn
- psutil (for memory tracking)
- mpi4py (for running in parallel structures)

The packages can be installed as follows. First, we recommend to create a virtual environment in Python3:

```create virtual env
# Create virtual environment
pyenv install 3.7.7
python3 -m venv ~/vqrc 
source ~/vqrc/bin/activate
```

Install the following packages for the basic functions of our implementations:
- Calculate the memory capacities of the higher-order quantum reservoir (HQR)
- Analyze the dynamics and bifurcation diagrams
- Perform NARMA tasks

```
# For running scripts in the nonlinear folder
pip3 install numpy matplotlib sklearn 
```

Install the following packages for chaos emulating tasks:

```
# For tracking memory
pip3 install psutil

# Install tensorflow to run LSTM, GRU scripts
pip install tensorflow==1.15.0

# For parallel structures in KuramotoSivashinskyGP64 task
sudo apt install libopenmpi-dev
pip3 install mpi4py
```

## Properties of higher-order quantum reservoir dynamics
To calculate the memory function MF(d) in the paper, run this command (see the running script run_hqrc_mem_func.sh for the detailed explanations):
```mfd
cd nonlinear/runscrips
sh run_hqrc_mem_func.sh
```

To calculate the memory capacity (MC) in the paper, run this command (see the running script run_hqrc_mem_capa.sh for the detailed explanations):
```mc
cd nonlinear/runscrips
sh run_hqrc_mem_capa.sh
```

To view the dynamics (the bifurcation diagrams and the time series), run this command (see the running script run_hqrc_view_states.sh for the detailed explanations):

```mc
cd nonlinear/runscrips
sh run_hqrc_view_states.sh
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
# Higher-Order Quantum Reservoir Computing

This repository is the official implementation of Higher-Order Quantum Reservoir Computing. 

[*] Quoc Hoan Tran and Kohei Nakajima, Higher-Order Quantum Reservoir Computing, Preprint at https://arxiv.org/abs/2006.08999

## Code structures

Our implementation includes two main parts (folders):

- nonlinear: The scripts to investigate the properties of the higher-order quantum reservoir (HQR) dynamics.
- chaos: The scripts to emulate Lorenz and KuramotoSivashinskyGP64 (KS64) chaotic systems. This part is integrated with the framework in [1,2] to make a fair comparision with echo state network (ESN), long short-term memory (LSTM), and gated recurrent unit (GRU) models.

## Requirements
The code requires the following libraries:
- python 3.6-3.7
- tensorflow 1.11.0-1.15.0
- numpy, matplotlib, sklearn
- psutil (for memory tracking)
- mpi4py (for running in parallel structures)

The packages can be installed as follows. First, we recommend to create a virtual environment in Python3 and use pyenv to manage python version:

```create virtual env
# Create virtual environment
pyenv install 3.7.7
python3 -m venv ~/vqrc 
source ~/vqrc/bin/activate
pyenv local 3.7.7
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

To calculate the quantum echo state property (QESP) index in the paper, run the following commands (see the running script run_calculate_esp.sh for the detailed explanations):

```mc
cd nonlinear/runscrips
sh run_calculate_esp.sh
```

To calculate the memory function MF(d) in the paper, run the following commands (see the running script run_hqrc_mem_func.sh for the detailed explanations):
```mfd
cd nonlinear/runscrips
sh run_hqrc_mem_func.sh
```

To calculate the memory capacity (MC) in the paper, run the following commands (see the running script run_hqrc_mem_capa.sh for the detailed explanations):

```mc
cd nonlinear/runscrips
sh run_hqrc_mem_capa.sh
```

To view the dynamics (the bifurcation diagrams and the time series), run the following commands (see the running script run_hqrc_view_states.sh for the detailed explanations):

```mc
cd nonlinear/runscrips
sh run_hqrc_view_states.sh
```

## NARMA tasks

To view the demo prediction for NARMA tasks, run the following commands:

```view-narma
cd nonlinear/runscrips
sh run_hqrc_view_narma.sh
```

To calculate the normalized mean-squred error (NMSE) loss for NARMA tasks with different settings of the parameters, run the following commands:

```nmse-narma
cd nonlinear/runscrips
sh run_hqrc_nmse_narma.sh
```

## Emulating Lorenz attractor

To generate simulation data and training/test dataset, run the following comands:
```generate data
cd chaos/Data/Lorenz3D
mkdir Data
mkdir Simulation_Data
python data_generation.py
python create_training_data
```

The above commands will create the dataset of 3x10^5 samples, and the first 10^5 samples are truncated to avoid initial transients. The remaining data are divided to a training and testing dataset of 10^5 samples each.

To train the HQR model(s) in the paper, run the following command (see Experiments/Lorenz3D/Local/HQRC_pinv.sh script for more detailed explanations):

```train
cd ../../../Methods
python3 RUN.py hqrc 
    --system_name Lorenz3D \
    --N 100000 \                # Number of time steps
    --N_used 10000 \            # Number of time steps used in training
    --noise_level 1 \           # Noise level added to the traning data
    --scaler MinMaxZeroOne \    # MinMax normalize the time series
    --nqrc 5 \                  # Number of QRs
    --alpha 0.1 \               # Connection strength
    --max_energy 2.0 \          # Max coupling energy
    --virtual_nodes 10 \        # Number of virtual nodes
    --tau 4.0 \                 # Interval between inputs
    --n_units 6 \               # Number of hidden units = num qubits
    --reg 1e-07 \               # Ridge parameter
    --dynamics_length 2000 \    # Transient time steps
    --it_pred_length 1000 \     # Iterative predicted length
    --n_tests 100 \             # Number of random tests
    --solver pinv               # Type of the solver in optimization
```
We also provide scripts (in Experiments/Lorenz3D/Local) to train the ESN, LSTM, and GRU models in the paper. These scripts are adapted from [1, 2] for our purpose.

We provide tuning hyperparameters for HQR, ESN, LSTM, and GRU models as follow.

HQR
| Hyperparameters (HQR)         | Explanation     |   Values       |
| ------------------ |---------------- | -------------- |
| reg  |  Ridge paramter for regularization   |   {1e-7, 1e-9, 1e-11}  |
| virtual_nodes  |  Number of virtual nodes   |   {10, 15, 20}  |
| alpha  |  Connection strength   |   {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}  |
| dynamics_length  |  Number of transitient time steps   |   200 if N_used=10^3, and 2000 if N_used=10^4,10^5  |
| noise_level  |  noise level added to the training data   |   {0.001, 0.005, 0.01}  |


ESN
| Hyperparameters (ESN)         | Explanation     |   Values       |
| ------------------ |---------------- | -------------- |
| degree  |  Degree of W_{h,h} matrix   |   {10}  |
| radius  |  Spectral radius of W_{h,h} matrix   |   {0.9}  |
| reg  |  Ridge paramter for regularization   |   {1e-5, 1e-7, 1e-9}  |
| n_nodes  |  Number of computational nodes   |   {80, 100, 120, 150, 500, 1000, 1500, 3000}  |
| dynamics_length  |  Number of transitient time steps   |   200 if N_used=10^3, and 2000 if N_used=10^4,10^5  |
| noise_level  |  noise level added to the training data   |   {0.001, 0.005, 0.01}  |

LSTM/GRU
| Hyperparameters (LSTM/GRU)         | Explanation     |   Values       |
| ------------------ |---------------- | -------------- |
| rnn_num_layers   |  Number of layers   |   {1,2,3}  |
| rnn_size_layers   |  Number of hidden units in each layer   |   {80, 100, 120, 150, 500, 1000, 1500, 3000}  |
| noise_level  |  noise level added to the training data   |   {0.001, 0.005, 0.01}  |
| batch_size   |  batch size   |   {32}  |
| sequence_length   |  Truncated backprop. length   |   {16}  |
| hidden_state_propagation_length   |  Number of warm-up time steps   |   100 if N_used=10^3, 1000 if N_used=10^4, and 2000 if N_used=10^5  |


## Emulating the KuramotoSivashinskyGP64 (KS64) model

To generate simulation data and training/test dataset, run the following comands:
```generate data
cd chaos/Data/KuramotoSivashinskyGP64
mkdir Data
python data_generation.py
```

The above commands will create the dataset of 24x10^4 samples, and the first 4x10^4 samples are truncated to avoid initial transients. The remaining data are divided to a training and testing dataset of 10^5 samples each.

To train the HQR model(s) in the paper, run the following command (see Experiments/KuramotoSivashinskyGP64/Local/HQRC_parallel.sh script for more detailed explanations):

```train
cd ../../../Methods
mpiexec -n 32 python3 RUN.py hqrc_parallel \
    --system_name KuramotoSivashinskyGP64 \
    --N 100000 \                    # Number of time steps
    --N_used 10000 \                # Number of time steps used in training
    --noise_level 1 \               # Noise level added to the traning data
    --scaler MinMaxZeroOne \        # MinMax normalize the time series
    --RDIM 64 \                     # Dim of the time series 
    --alpha 0.1 \                   # Connection strength
    --max_energy 2.0 \              # Max coupling energy
    --fix_coupling 1 \              # Ising model g=1.0, h_ij in [-0.5, 0.5]
    --virtual_nodes 10 \            # Number of virtual nodes
    --tau 4.0 \                     # Interval between inputs
    --n_units 6 \                   # Number of hidden units = num qubits
    --reg 1e-07 \                   # Ridge parameter
    --nqrc 10 \                     # Number of QRs
    --dynamics_length 2000 \        # Transient time steps
    --it_pred_length 400 \          # Predicted length
    --n_tests 100 \                 # Number of tests
    --solver pinv \                 # Type of the solver in optimization
    --augment 1 \                   # Augment the hidden states
    --n_groups 32 \                 # Number of groups 
    --group_interaction_length 4 \  # Interaction length between groups
```

We also provide scripts (in Experiments/Lorenz3D/Local) to train the ESN, LSTM, and GRU models in the paper with the same parallel structures. These scripts are adapted from [1, 2] for our purpose.

## Evaluation

To evaluate the prediction performance on chaos emulating tasks, run the following commands:

```eval
cd Postprocess
python sort_model_test_vpt.py --sysname Lorenz3D --tag N_used_10000-
python sort_model_test_vpt.py --sysname KuramotoSivashinskyGP64 --tag N_used_10000-
```

The above commands will produce the text file, which sorts all models from the highest average VPT to the lowest average VPT (see folder Eval_Figures for these files).

You can download the evaluation files of our experiments:
- [My evaluation files for Lorenz3D](https://www.dropbox.com/s/g9589zdalk8qm8c/Evaluation_Data_Lorenz3D.zip?dl=0)
- [My evaluation files for KuramotoSivashinskyGP64](https://www.dropbox.com/s/qrx5ojttn05zxko/Evaluation_Data_KuramotoSivashinskyGP64.zip?dl=0)

To reproduce the figures of Lorenz system in the manuscript, run the following commands:
```eval
cd Postprocess

# Training time steps = 10^3 
python plot_series_compare_Lorenz3D.py --tidx 2 --used 0 

# Training time steps = 10^4 
python plot_series_compare_Lorenz3D.py --tidx 2 --used 1

# Training time steps = 10^5
python plot_series_compare_Lorenz3D.py --tidx 2 --used 2 

# Compare the models when varying the number of time steps for training
python plot_series_compare_Lorenz3D.py --tidx 3 --used 3

# Compare performances when varying the connection strength
python plot_series_compare_Lorenz3D.py --tidx 2 --used 4 
```

To reproduce the figures of KuramotoSivashinskyGP64 in the manuscript, run the following commands:
```eval
cd Postprocess
python plot_contour_compare_KS.py
```
## Results

See folders "Results/Lorenz3D/Eval_Figures" and "Results/KuramotoSivashinskyGP64/Eval_Figures".


## References
[1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., Backpropagation algorithms and Reservoir Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal dynamics. Neural Networks (2020), doi: https://doi.org/10.1016/j.neunet.2020.02.016.

[2] CSElab ETH Zurich, Recurrent neural network architectures based on backpropagation and reservoir computing for forecasting high-dimensional chaotic dynamical systems, https://github.com/pvlachas/RNN-RC-Chaos
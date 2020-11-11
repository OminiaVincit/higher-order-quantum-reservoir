#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
    Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
    Adapted to Higher-order quantum reservoir computing by Quoc Hoan Tran

    Implemented in the framework created by Vlachas Pantelis, CSE-lab, ETH Zurich
        https://github.com/pvlachas/RNN-RC-Chaos
        [1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., 
        Backpropagation algorithms and Reservoir Computing in Recurrent Neural Networks 
        for the forecasting of complex spatiotemporal dynamics. Neural Networks (2020), 
        doi: https://doi.org/10.1016/j.neunet.2020.02.016.

"""
#!/usr/bin/env python
import sys
import socket
import os

global_params = lambda:0
global_params.cluster = 'local'

if global_params.cluster == 'local':
    print("## CONFIG: RUNNING IN LOCAL REPOSITORY.")
    config_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(os.path.dirname(config_path))

print("PROJECT PATH={}".format(project_path))

global_params.global_utils_path = "./Models/Utils"

global_params.saving_path = project_path + "/Results/{:s}"
global_params.project_path = project_path

global_params.training_data_path = project_path + "/Data/{:s}/Data/training_data_N{:}.pickle"
global_params.testing_data_path = project_path + "/Data/{:s}/Data/testing_data_N{:}.pickle"

# PATH TO LOAD THE PYTHON MODELS
global_params.py_models_path = "./Models/{:}"

# PATHS FOR SAVING RESULTS OF THE RUN
global_params.model_dir = "/Trained_Models/"
global_params.fig_dir = "/Figures/"
global_params.results_dir = "/Evaluation_Data/"
global_params.logfile_dir = "/Logfiles/"


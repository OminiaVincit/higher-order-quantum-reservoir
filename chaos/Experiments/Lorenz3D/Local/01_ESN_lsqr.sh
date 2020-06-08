#!/bin/bash

cd ../../../Methods


for R in 1e-3 1e-4 1e-5 1e-6
do
python3 RUN.py esn \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 1000 \
--RDIM 1 \
--scaler Standard \
--approx_reservoir_size 5000 \
--degree 10 \
--radius 0.6 \
--sigma_input 1 \
--regularization $R \
--dynamics_length 200 \
--iterative_prediction_length 500 \
--noise_level 0 \
--num_test_ICS 101 \
--solver lsqr \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done

#!/bin/bash
#export OMP_NUM_THREADS=12

cd ../../../Methods

for UNITS in 500 1000
do
for BETA in 1e-7
do
python3 RUN.py esn \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM 1 \
--noise_level 1 \
--scaler Standard \
--approx_reservoir_size $UNITS \
--degree 10 \
--radius 0.9 \
--sigma_input 1 \
--regularization $BETA \
--dynamics_length 2000 \
--iterative_prediction_length 1000 \
--num_test_ICS 2 \
--solver pinv \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
done





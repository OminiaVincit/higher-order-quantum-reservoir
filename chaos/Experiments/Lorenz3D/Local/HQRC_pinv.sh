#!/bin/bash

cd ../../../Methods

for ALPHA in 0.0
do
for REG in 1e-7
do
python3 RUN.py hqrc \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 1000 \
--RDIM 1 \
--noise_level 5 \
--scaler MinMaxZeroOne \
--trans 20.0 \
--ratio 200.0 \
--nqrc 5 \
--layer_strength $ALPHA \
--max_coupling_energy 1.0 \
--virtual_nodes 5 \
--tau_delta 2.0 \
--hidden_unit_count 6 \
--regularization $REG \
--dynamics_length 200 \
--iterative_prediction_length 500 \
--num_test_ICS 2 \
--solver pinv \
--augment 1 \
--norm_every 10 \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
done






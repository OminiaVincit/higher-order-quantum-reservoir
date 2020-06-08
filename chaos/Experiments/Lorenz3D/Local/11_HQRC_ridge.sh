#!/bin/bash

cd ../../../Methods



python3 RUN.py hqrc \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 1000 \
--RDIM 1 \
--noise_level 1 \
--scaler MinMaxZeroOne \
--nqrc 1 \
--layer_strength 0.0 \
--max_coupling_energy 1.0 \
--virtual_nodes 20 \
--tau_delta 2.0 \
--hidden_unit_count 6 \
--regularization 1e-5 \
--dynamics_length 200 \
--iterative_prediction_length 500 \
--num_test_ICS 1 \
--solver sag \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5






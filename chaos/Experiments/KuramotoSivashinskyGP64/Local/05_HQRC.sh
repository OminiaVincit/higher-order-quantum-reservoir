#!/bin/bash

cd ../../../Methods

for RDIM in 64
do
for ALPHA in 0.5 0.0
do
for RE in 1e-5 1e-7 1e-9
do
python3 RUN.py hqrc \
--mode all \
--display_output 1 \
--system_name KuramotoSivashinskyGP64 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM $RDIM \
--noise_level 5 \
--scaler MinMaxZeroOne \
--scale_input 1.0 \
--nqrc $RDIM \
--layer_strength $ALPHA \
--max_coupling_energy 1.0 \
--virtual_nodes 50 \
--tau_delta 2.0 \
--hidden_unit_count 6 \
--regularization $RE \
--solver pinv \
--dynamics_length 2000 \
--iterative_prediction_length 1000 \
--num_test_ICS 2
done
done
done

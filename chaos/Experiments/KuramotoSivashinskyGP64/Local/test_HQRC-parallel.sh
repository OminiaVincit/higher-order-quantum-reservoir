#!/bin/bash

cd ../../../Methods

for NR in 32
do
for RE in 1e-5
do
for UL in 0
do
mpiexec -n $NR python3 RUN.py hqrc_parallel \
--mode all \
--display_output 1 \
--system_name KuramotoSivashinskyGP64 \
--write_to_log 1 \
--N 100000 \
--N_used 1000 \
--RDIM 32 \
--noise_level 5 \
--scaler MinMaxZeroOne \
--trans 10.0 \
--ratio 20.0 \
--scale_input 1.0 \
--layer_strength 0.0 \
--max_coupling_energy 1.0 \
--virtual_nodes 1 \
--tau_delta 2.0 \
--hidden_unit_count 6 \
--regularization $RE \
--solver pinv \
--norm_every 0 \
--augment 0 \
--nqrc 3 \
--num_parallel_groups $NR \
--parallel_group_interaction_length 1 \
--dynamics_length 200 \
--iterative_prediction_length 400 \
--iterative_update_length $UL \
--num_test_ICS 1
done
done
done

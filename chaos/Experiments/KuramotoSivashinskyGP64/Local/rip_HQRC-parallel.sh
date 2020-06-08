#!/bin/bash

cd ../../../Methods

for NR in 32
do
for RE in 1e-07
do
for ALPHA in 0.0
do
for V in 25 20 15 10
do
mpiexec -n $NR python3 RUN.py hqrc_parallel \
--mode all \
--display_output 1 \
--system_name KuramotoSivashinskyGP64 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM 64 \
--noise_level 1 \
--scaler MinMaxZeroOne \
--trans 10.0 \
--ratio 20.0 \
--scale_input 1.0 \
--layer_strength $ALPHA \
--max_coupling_energy 2.0 \
--fix_coupling 1 \
--virtual_nodes $V \
--tau_delta 4.0 \
--hidden_unit_count 6 \
--regularization $RE \
--solver pinv \
--norm_every 0 \
--augment 1 \
--nqrc 10 \
--num_parallel_groups $NR \
--parallel_group_interaction_length 4 \
--dynamics_length 2000 \
--iterative_prediction_length 400 \
--iterative_update_length 0 \
--num_test_ICS 100
done
done
done
done

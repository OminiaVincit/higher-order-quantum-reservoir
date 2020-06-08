#!/bin/bash
cd ../../../Methods
for AR in 80 100 120 150 500 1000 1500 3000
do
for NR in 32
do
for RE in 1e-07 1e-05
do
mpiexec -n $NR python3 RUN.py esn_parallel \
--mode all \
--display_output 0 \
--system_name KuramotoSivashinskyGP64 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM 64 \
--scaler Standard \
--approx_reservoir_size $AR \
--degree 10 \
--radius 0.6 \
--sigma_input 1 \
--regularization $RE \
--dynamics_length 2000 \
--num_parallel_groups $NR \
--parallel_group_interaction_length 4 \
--iterative_prediction_length 400 \
--num_test_ICS 100 \
--noise_level 1
done
done
done


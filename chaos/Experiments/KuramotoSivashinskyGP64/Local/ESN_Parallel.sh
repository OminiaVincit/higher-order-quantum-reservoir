#!/bin/bash
#export OMP_NUM_THREADS=12

cd ../../../Methods
for AR in 150 500
do
for NR in 32
do
for BETA in 1e-07 1e-05
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
--n_nodes $AR \
--degree 10 \
--radius 0.6 \
--sigma_input 1 \
--reg $BETA \
--dynamics_length 2000 \
--n_groups $NR \
--group_interaction_length 4 \
--it_pred_length 400 \
--n_tests 2 \
--noise_level 1
done
done
done


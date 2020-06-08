#!/bin/bash

cd ../../../Methods

for RDIM in 64
do
for ALPHA in 0.5 0.0
do
for RE in 1e-7 1e-5 1e-9
do
for AU in 0 1
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
--alpha $ALPHA \
--max_energy 1.0 \
--virtual_nodes 50 \
--tau 2.0 \
--n_units 6 \
--regularization $RE \
--solver pinv \
--augment $AU \
--dynamics_length 2000 \
--iterative_prediction_length 1000 \
--num_test_ICS 1
done
done
done
done

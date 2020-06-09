#!/bin/bash
# Script to emulate chaos by using hqrc
# 
    # --N_used 10000 \ # Number of time steps in training
    # --RDIM 1 \ # Dim of the input data
    # --noise_level 1 \ # Noise level added to the traning data
    # --scaler MinMaxZeroOne \
    # --nqrc 5 \ # Number of QRs
    # --alpha $ALPHA \ # Connection strength
    # --max_energy 2.0 \ # Max coupling energy
    # --fix_coupling 1 \ # Use special type of the Ising model g=1.0, h_ij in [-0.5, 0.5]
    # --virtual_nodes $V \ # Number of virtual nodes
    # --tau 4.0 \ # Interval between inputs
    # --n_units 6 \ # Number of hidden units =qubits in our setting
    # --regularization $BETA \ # Ridge parameter
    # --dynamics_length 2000 \ # Transient time steps
    # --iterative_prediction_length 400 \ # Predicted length
    # --num_test_ICS 2 \ # Number of tests
    # --solver pinv \ # Ridge by pseudo inverse
    # --augment 0 \ # Augment the hidden states
    # --num_parallel_groups 32 \ # Number of groups 
    # --parallel_group_interaction_length 4 \ # Interaction length between groups
    #
# End of the script to emulate chaos by using hqrc
# 
cd ../../../Methods
#export OMP_NUM_THREADS=12

for NR in 32
do
for BETA in 1e-07
do
for ALPHA in 0.0
do
for V in 10
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
--alpha $ALPHA \
--max_energy 2.0 \
--fix_coupling 1 \
--virtual_nodes $V \
--tau 4.0 \
--n_units 6 \
--regularization $BETA \
--solver pinv \
--norm_every 0 \
--augment 1 \
--nqrc 10 \
--num_parallel_groups $NR \
--parallel_group_interaction_length 4 \
--dynamics_length 2000 \
--iterative_prediction_length 400 \
--iterative_update_length 0 \
--num_test_ICS 2
done
done
done
done

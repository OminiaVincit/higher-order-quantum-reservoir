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
    # --virtual_nodes $V \ # Number of virtual nodes
    # --tau 4.0 \ # Interval between inputs
    # --n_units 6 \ # Number of hidden units =qubits in our setting
    # --reg $BETA \ # Ridge parameter
    # --dynamics_length 2000 \ # Transient time steps
    # --it_pred_length 1000 \ # Predicted length
    # --n_tests 2 \ # Number of tests
    # --solver pinv \ # Ridge by pseudo inverse
    # --augment 0 \ # Augment the hidden states
# End of the script to emulate chaos by using hqrc
# 
cd ../../../Methods
export OMP_NUM_THREADS=12

for V in 10
do
for ALPHA in 0.1
do
for BETA in 1e-7
do
python3 RUN.py hqrc_innate \
    --mode all \
    --display_output 1 \
    --system_name Lorenz3D \
    --write_to_log 1 \
    --N 100000 \
    --N_used 1000 \
    --RDIM 1 \
    --noise_level 1 \
    --output_noise 1e-6\
    --innate_learning_rate 10.0\
    --innate_learning_loops 10\
    --scale_input 0.2\
    --scaler MinMaxZeroOne \
    --nqrc 5 \
    --alpha $ALPHA \
    --max_energy 1.0 \
    --virtual_nodes $V \
    --tau 2.0 \
    --n_units 6 \
    --reg $BETA \
    --dynamics_length 200 \
    --it_pred_length 100 \
    --n_tests 2 \
    --solver pinv \
    --augment 0
done
done
done
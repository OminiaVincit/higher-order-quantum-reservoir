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

NQR=2
W=1.0
TAU=4.0
UNITS=5
USECORR=0

for V in 1
do
for GAMMA in 0.0
do
for BETA in 1e-7
do
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
    --nqrc $NQR \
    --gamma $GAMMA \
    --non_diag_var $W \
    --virtual_nodes $V \
    --use_corr $USECORR \
    --tau $TAU \
    --n_units $UNITS \
    --reg $BETA \
    --dynamics_length 200 \
    --it_pred_length 100 \
    --n_tests 2 \
    --solver pinv \
    --augment 0
done
done
done
#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/eff_dim.py
LENGTH=100000
BUFFER=90000

#LENGTH=11000
#BUFFER=10000

SAVE=../../../data/hqrc/dynamics_sinwave/eff
INPUT_FILE=../data/sin_input_T_50.txt

QR=5
PROC=100
CONST=0
INT=0.01
TP=5
OP='X'

RSFILE=../../data/qrc_topo/res_eff_dim/phase_trans_nqr_1_V_1_tau_10.0_nondiag_2.0_tp_1_\_rsd_0_esp_id_99_len_$LENGTH.binaryfile

for RD in 0
do
python $BIN --gamma $GAM --type_op $OP  --input_file $INPUT_FILE --randseed $RD --type_input $TP --interval $INT --savedir $SAVE --length $LENGTH --buffer $BUFFER --nqrc $QR --nproc $PROC
#python $BIN --trials $TRIALS --type_input $TP --interval $INT --savedir $RSFILE --length $LENGTH --buffer $BUFFER --nqrc $QR --nproc $PROC
done
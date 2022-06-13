#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/ESP_index_heat_2D.py
LENGTH=10000
BUFFER=9000

#LENGTH=2000
#BUFFER=1000

TRIALS=1


BASE='rand_input_2001000'
SAVE=../../../data/hqrc/$BASE/esp_2D

#INPUT_FILE=../data/sin_input_T_50.txt
INPUT_FILE=../data/$BASE.txt

QR=1
PROC=100
CONST=0
INT=0.05
TP=5
OP='X'

RSFILE=$SAVE/heat_phase_trans_nqr_1_V_1_tau_10.0_nondiag_2.0_op_X_tp_5_trials_$TRIALS\_rsd_0_esp_id_81_len_$LENGTH.binaryfile

for RD in 0
do
python $BIN --type_op $OP  --input_file $INPUT_FILE --randseed $RD --trials $TRIALS --type_input $TP --interval $INT --savedir $SAVE --length $LENGTH --buffer $BUFFER --nqrc $QR --nproc $PROC
#python $BIN --trials $TRIALS --type_input $TP --interval $INT --savedir $RSFILE --length $LENGTH --buffer $BUFFER --nqrc $QR --nproc $PROC
done
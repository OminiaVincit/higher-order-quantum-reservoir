#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/ESP_index.py
LENGTH=10000
BUFFER=9000

LENGTH=2000
BUFFER=1000

TRIALS=10


BASE='rand_input_2001000'
SAVE=../../../data/hqrc/$BASE/esp

#INPUT_FILE=../data/sin_input_T_50.txt
INPUT_FILE=../data/$BASE.txt

QR=1
PROC=111
CONST=0
INT=0.01
TP=5
OP='X'

RSFILE=../../data/qrc_topo/res_esp_index/phase_trans_nqr_1_V_1_tau_10.0_nondiag_2.0_tp_1_trials_$TRIALS\_rsd_0_esp_id_99_len_$LENGTH.binaryfile

for RD in 0 1 2 3 4 5 6 7 8 9
do
for GAM in -1.0
do
for W in 0.0 0.1 1.0 10.0 100.0
do
python $BIN --non_diag_var $W --gamma $GAM --type_op $OP  --input_file $INPUT_FILE --randseed $RD --trials $TRIALS --type_input $TP --interval $INT --savedir $SAVE --length $LENGTH --buffer $BUFFER --nqrc $QR --nproc $PROC
#python $BIN --trials $TRIALS --type_input $TP --interval $INT --savedir $RSFILE --length $LENGTH --buffer $BUFFER --nqrc $QR --nproc $PROC
done
done
done
#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states_angle.py

# LENGTH=101000
LENGTH=10000
BG=9000
ED=9200

# LENGTH=1000
# ED=1000
# BG=800

SAVE=../../../data/hqrc/dynamics_sinwave
INPUT_FILE=../data/sin_input_T_50.txt

QR=5
PROC=100
CONST=0
INT=0.01

#RSFILE=../../data/qrc_topo/dynamics_binary_long/phase_trans_nqr_1_V_1_tp_1_states_id_81_len_101000.binaryfile

for OP in 'X'
do
for TP in 5
do
for GAM in 1.0
do
for RD in 0
do
python $BIN --gamma $GAM --type_op $OP --input_file $INPUT_FILE --randseed $RD --type_input $TP --interval $INT --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --nproc $PROC
done
done
done
done
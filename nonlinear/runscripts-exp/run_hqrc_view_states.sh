#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states.py
LENGTH=500
BG=400
ED=500
SAVE=../../../data/hqrc/dynamics_scale_input
QR=5
PROC=100
CONST=0
INT=0.05
SM=3
CB=0
SP=1.0
TP=0

for SC in 1.0
do
for MS in 1
do
for SG in 1.0
do
for ALPHA in 0.1 0.5
do
python $BIN --scale_input $SC --type_input $TP --mask_input $MS --combine_input $CB --sigma_input $SG --sparsity $SP --nonlinear $SM --interval $INT --const $CONST --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --strength $ALPHA --nproc $PROC
done
done
done
done
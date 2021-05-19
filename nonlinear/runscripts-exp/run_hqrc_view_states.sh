#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states.py
LENGTH=5000
BG=4000
ED=5000
SAVE=../../../data/hqrc/dynamics_sigmoid
QR=5
PROC=100
CONST=0
INT=0.05
SM=0
SP=1.0
SG=1.0

for SG in 1.0
do
for ALPHA in 0.1 0.05
do
python $BIN --sigma_input $SG --sparsity $SP --nonlinear $SM --interval $INT --const $CONST --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --strength $ALPHA --nproc $PROC
done
done

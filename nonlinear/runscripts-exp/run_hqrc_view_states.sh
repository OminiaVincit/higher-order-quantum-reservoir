#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states.py
LENGTH=5000
BG=4000
ED=5000
SAVE=../../../data/hqrc/dynamics_softmax
QR=5
PROC=100
CONST=0
INT=0.05
SM=0
for SP in 0.1 0.5 0.9
do
for ALPHA in 0.5
do
python $BIN --sparsity $SP --softmax $SM --interval $INT --const $CONST --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --strength $ALPHA --nproc $PROC
done
done

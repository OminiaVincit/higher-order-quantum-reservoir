#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states.py
LENGTH=500
BG=400
ED=500
SAVE=../../../data/hqrc/dynamics_norm_feed_bw_input
#SAVE=zoro/data/hqrc/dynamics_norm_feed_bw_input

QR=5
PROC=100
CONST=0
INT=0.05
SM=0
CB=0
SP=1.0
TP=0
TS=0.0

for SC in 1.0
do
for MS in 1
do
for SG in 1.0
do
for ALPHA in 1.0
do
python $BIN --trans_input $TS --scale_input $SC --type_input $TP --mask_input $MS --combine_input $CB --sigma_input $SG --sparsity $SP --nonlinear $SM --interval $INT --const $CONST --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --strength $ALPHA --nproc $PROC
done
done
done
done
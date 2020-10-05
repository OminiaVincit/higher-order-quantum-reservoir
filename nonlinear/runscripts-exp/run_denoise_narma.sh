#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/multi_denoise_innate.py
TRAIN=2000
VAL=2000
T=2000
N=10
TRAIN=1
RATE=0.0
SAVE=../denoise_rmse
TAUS='1.0,2.0,4.0'

for ORDER in '10'
do
python $BIN --savedir $SAVE --trainloops $TRAIN --learning_rate $RATE --virtuals 1 --scale_input 0.2 --ntrials $N --orders $ORDER --taudelta $TAUS --ranseed 0
done


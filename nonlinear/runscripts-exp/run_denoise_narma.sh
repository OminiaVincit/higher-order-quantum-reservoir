#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/multi_denoise_innate.py
TRAIN=2000
VAL=2000
T=2000
N=10
TRAIN=20
RATE=10.0
SAVE=../de-narma3

O1='5,10'
O2='15,20'
for ORDER in $O1 $O2
do
python $BIN --savedir $SAVE --trainloops $TRAIN --learning_rate $RATE --virtuals 1 --scale_input 0.2 --ntrials $N --orders $ORDER --taudelta 4.0 --ranseed 0
done


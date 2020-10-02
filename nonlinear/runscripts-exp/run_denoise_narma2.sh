#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/denoise_innate_hqrc.py
TRAIN=2000
VAL=2000
T=2000
N=10
TRAIN=20
RATE=10.0
SAVE=../de-narma2
ORDER=10

for NOISE in 1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6
do
python $BIN --savedir $SAVE --trainloops $TRAIN --learning_rate $RATE --virtuals 1 --scale_input 0.2 --ntrials $N --order $ORDER --taudelta 4.0 --ranseed 0 --noise $NOISE
done

for NOISE in 1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5
do
python $BIN --savedir $SAVE --trainloops $TRAIN --learning_rate $RATE --virtuals 1 --scale_input 0.2 --ntrials $N --order $ORDER --taudelta 4.0 --ranseed 0 --noise $NOISE
done

for NOISE in 1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4
do
python $BIN --savedir $SAVE --trainloops $TRAIN --learning_rate $RATE --virtuals 1 --scale_input 0.2 --ntrials $N --order $ORDER --taudelta 4.0 --ranseed 0 --noise $NOISE
done

for NOISE in 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2
do
python $BIN --savedir $SAVE --trainloops $TRAIN --learning_rate $RATE --virtuals 1 --scale_input 0.2 --ntrials $N --order $ORDER --taudelta 4.0 --ranseed 0 --noise $NOISE
done
#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/narma_hqrc.py
TRAIN=2000
VAL=2000
T=2000
UNITS=5
SOLVER='ridge_pinv'

SAVE=/data/zoro/hqrc/narma_nrmse_tau
QR='5'
N=10
#TAUS='0.0,1.0,2.0,3.0,4.0'
#TAUS=\'-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7\'
TAUS='default'
for V in 15 20 10
do
for DEEP in 0
do
for ALPHA in 0.0 0.5 0.9
do
python $BIN --deep $DEEP --units $UNITS --strengths $ALPHA --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAUS
done
done
done
#SOLVER='linear_pinv'
#python $BIN --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
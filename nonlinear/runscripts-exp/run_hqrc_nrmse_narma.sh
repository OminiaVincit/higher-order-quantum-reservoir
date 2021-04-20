#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/narma_hqrc.py
TRAIN=2000
VAL=2000
T=2000
UNITS=5

SAVE=../../../data/hqrc/narma_nrmse_tau
QR='5'
N=10
V=20
#TAU='0.0,1.0,2.0,3.0,4.0'
TAUS=\'-3,-2,-1,0,1,2,3,4,5,6,7\'
ALPHA='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'

SOLVER='ridge_pinv'
python $BIN --units $UNITS --strengths $ALPHA --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU

#SOLVER='linear_pinv'
#python $BIN --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
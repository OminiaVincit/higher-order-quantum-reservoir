#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/narma_hqrc.py
TRAIN=2000
VAL=2000
T=2000
UNITS=5

SAVE=../../../data/hqrc/narma_nrmse_alpha_tau_8
QR='1,2,3,4,5'
N=10
V=20
#TAU='0.0,1.0,2.0,3.0,4.0'
TAUS='3.0' #2**x for x in TAUS
ALPHA='0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85'

SOLVER='ridge_pinv'
for DEEP in 0 1
do
python $BIN --deep $DEEP --units $UNITS --strengths $ALPHA --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAUS
done
#SOLVER='linear_pinv'
#python $BIN --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
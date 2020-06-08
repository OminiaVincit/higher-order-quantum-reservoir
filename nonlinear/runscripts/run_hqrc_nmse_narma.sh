#!/usr/bin/bash
export OMP_NUM_THREADS=1

BIN=../source/narma_hqrc.py
TRAIN=2000
VAL=2000
T=2000

SAVE=../test_narma_nmse
QR='5'
N=2
V=20
TAU='0.0,1.0,2.0'

SOLVER='ridge_pinv'
python $BIN --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU

SOLVER='linear_pinv'
python $BIN --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU

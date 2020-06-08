#!/usr/bin/bash
# Script to view predicted time series of NARMA tasks
export OMP_NUM_THREADS=1

BIN=../source/view_narma.py
TRAIN=2000
VAL=2000
T=2000

SAVE=../test_narma
QR=5
N=10
V=20
TAU=2.0
SOLVER='ridge_pinv'

python $BIN --solver $SOLVER --plot 0 --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
python $BIN --solver $SOLVER  --plot 1 --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU

SOLVER='linear_pinv'

python $BIN --solver $SOLVER  --plot 0 --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
python $BIN --solver $SOLVER  --plot 1 --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
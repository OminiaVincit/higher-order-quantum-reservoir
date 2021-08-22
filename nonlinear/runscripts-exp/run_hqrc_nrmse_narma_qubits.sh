#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/narma_hqrc.py
TRAIN=2000
VAL=2000
T=2000
UNITS=5
DEEP=0
SAVE=../../../data/hqrc/narma_nrmse_oneinput_tau_8_qr
QR='5'
N=10
#TAU='0.0,1.0,2.0,3.0,4.0'
TAUS='3.0'
#ALPHA='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.999,0.9999,0.99999,1.0'
ALPHA='0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0'

CB_INPUT=2
MS_INPUT=0
TP_INPUT=0
NONLINEAR=0
BNORM=0

ORDERS='5,10,15,20'
SOLVER='linear_pinv'

for QR_INPUT in 4 3 2 1
do
for SIG_INPUT in 1.0
do
for V in 1
do
python $BIN --qr_input $QR_INPUT --bnorm $BNORM --orders $ORDERS --sigma_input $SIG_INPUT --nonlinear $NONLINEAR --type_input $TP_INPUT --mask_input $MS_INPUT --combine_input $CB_INPUT --deep $DEEP --units $UNITS --strengths $ALPHA --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAUS
done
done
done
#SOLVER='linear_pinv'
#python $BIN --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
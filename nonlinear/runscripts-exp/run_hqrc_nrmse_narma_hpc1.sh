#!/usr/bin/bash
# Script to calculate NMSE for NARMA task
export OMP_NUM_THREADS=1

BIN=../source/narma_hqrc.py
TRAIN=2000
VAL=2000
T=2000
UNITS=5

SAVE=/data/zoro/hqrc/trans_del_narma_pretrain
QR='5'
#ORDERS='2,5,10,15,20'
ORDERS='5'

N=1

#TAU='0.0,1.0,2.0,3.0,4.0,5.0,6.0'
TAUS=\'-2,-1,0,1,2,3,4,5,6,7\'
#TAUS='4.0' #2**x for x in TAUS
#ALPHA='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.999,0.9999,0.99999'
ALPHA='0.9'
SOLVER='linear_pinv'
SAVE_MODEL=0
LOAD_MODEL=0
LOAD_ORDER=2
CB_INPUT=1
VIEW_DYN=1
NONLINEAR=0

for V in 1
do
for DEEP in 0
do
python $BIN --nonlinear $NONLINEAR --view_dynamic $VIEW_DYN --combine_input $CB_INPUT --save_model $SAVE_MODEL --load_model $LOAD_MODEL --load_order $LOAD_ORDER --orders $ORDERS --deep $DEEP --units $UNITS --strengths $ALPHA --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAUS
done
done
#SOLVER='linear_pinv'
#python $BIN --solver $SOLVER --savedir $SAVE --trainlen $TRAIN --vallen $VAL --transient $T --nqrc $QR --ntrials $N --virtuals $V --taudelta $TAU
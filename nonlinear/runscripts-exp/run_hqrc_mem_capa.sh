#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/mc_hqrc.py
N=2
J=1.0
TASK=qrc_stm
SAVE=../test_mc
QR=5
PROC=24

TAUS=\'-1,0,3\' # The parameters for tau is 2**x for x in TAUS
STRENGTHS='0.1,0.5,0.9'
VS='1,5'
QRS='5'

MIND=0
MAXD=20
INT=1

for SOLVER in 'ridge_pinv' 'linear_pinv'
do
    python $BIN --solver $SOLVER --coupling $J --taudeltas $TAUS --taskname $TASK --layers $QRS --strengths $STRENGTHS --virtuals $VS --ntrials $N --nproc $PROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
done
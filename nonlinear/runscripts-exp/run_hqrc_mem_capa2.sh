#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/mc_hqrc.py
N=10
J=1.0
TASK=qrc_stm
SAVE=../../../data/hqrc/memcapa
QR=5
PROC=101

TAUS=\'-1,4\' # The parameters for tau is 2**x for x in TAUS
STRENGTHS='1.0'
VS='1,5,10,15,25'
QRS='5'

MIND=0
MAXD=200
INT=1

for SOLVER in 'ridge_pinv'
do
    python $BIN --solver $SOLVER --coupling $J --taudeltas $TAUS --taskname $TASK --layers $QRS --strengths $STRENGTHS --virtuals $VS --ntrials $N --nproc $PROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
done
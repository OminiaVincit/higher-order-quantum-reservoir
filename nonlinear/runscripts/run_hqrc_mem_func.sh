#!/usr/bin/bash
export OMP_NUM_THREADS=1

BIN=../source/mf_hqrc.py
N=2
J=1.0
TASK=qrc_stm
SAVE=../test_mf
QR=5
PROC=24
SOLVER='ridge_pinv'
TAUS=\'-1,0,3\'
MIND=0
MAXD=20
INT=1

for a in 0.5
do
for V in 5 
do
    python $BIN --solver $SOLVER --coupling $J --taudeltas $TAUS --taskname $TASK --nqrc $QR --strength $a --virtuals $V --ntrials $N --nproc $PROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
done
done

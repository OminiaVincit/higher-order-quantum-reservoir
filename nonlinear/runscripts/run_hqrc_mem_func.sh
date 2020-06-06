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

for a in 0.5
do
for V in 5 
do
    python $BIN --solver $SOLVER --coupling $J --taskname $TASK --nqrc $QR --strength $a --virtuals $V --ntrials $N --nproc $PROC --mind 0 --maxd 20 --savedir $SAVE
done
done

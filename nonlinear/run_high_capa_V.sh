#!/usr/bin/bash
export OMP_NUM_THREADS=1

BIN=source/mem_capacity_highorder_tau_change.py
N=10
J=1.0
TASK=qrc_stm
SAVE=rescapa_high_V
L=5
PROC=80

for a in 0.5
do
    for V in 5 10 
    do
        python $BIN --coupling $J --plot 0 --taskname $TASK --layers $L --strength $a --virtuals $V --ntrials $N --nproc $PROC --maxd 200 --savedir $SAVE
    done
done

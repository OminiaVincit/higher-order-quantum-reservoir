#!/usr/bin/bash
export OMP_NUM_THREADS=1

BIN=source/mem_capacity_highorder_tau_change.py
N=10
V=1
TASK=qrc_stm
SAVE=rescapa_high_coupling
L=5
PROC=90

for a in 0.9 0.0 0.5
do
    for J in 2.0 0.5 4.0 0.25 8.0 0.125 1.0 16.0 0.0625 
    do
        python $BIN --coupling $J --plot 0 --taskname $TASK --layers $L --strength $a --ntrials $N --nproc $PROC --maxd 200 --savedir $SAVE
    done
done
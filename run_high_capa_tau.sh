#!/usr/bin/bash
BIN=source/mem_capacity_highorder_tau_change.py
N=10
V=1
TASK=qrc_stm
SAVE=rescapa_high_stm2
L=2,3,4,5
python $BIN --plot 1 --strength 0.5 --ntrials $N --virtuals $V --taskname $TASK --layers $L --nproc 51 --maxd 200 --savedir $SAVE
python $BIN --plot 1 --strength 0.9 --ntrials $N --virtuals $V --taskname $TASK --layers $L --nproc 51 --maxd 200 --savedir $SAVE

#!/usr/bin/bash
BIN=source/mem_capacity_highorder_tau_change.py
N=10
V=1
TASK=qrc_stm
SAVE=rescapa_high_strength
L=5
python $BIN --plot 0 --strength 0.3 --ntrials $N --virtuals $V --taskname $TASK --layers $L --nproc 51 --maxd 200 --savedir $SAVE
python $BIN --plot 0 --strength 0.7 --ntrials $N --virtuals $V --taskname $TASK --layers $L --nproc 51 --maxd 200 --savedir $SAVE
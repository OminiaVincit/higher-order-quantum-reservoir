#!/usr/bin/bash
BIN=source/lyapuv_highorder.py
N=10
S=1e-2
L=5
SDIR=res_high_lyp
BUFFER=1000
LENGTH=5000
V=1

ALPHA=0.1,0.5,0.9

for J in 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0
do
  python $BIN --strengths $ALPHA --savedir $SDIR --buffer $BUFFER --length $LENGTH --coupling $J --savedir $SDIR --ntrials $N --initial_distance $S --virtuals $V --layers $L
done


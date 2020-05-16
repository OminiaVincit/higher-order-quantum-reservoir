#!/usr/bin/bash
BIN=source/esp_highorder.py
N=10
S=10
L=5
SDIR=res_high_echo6
BUFFER=1000
LENGTH=2000
V=1

ALPHA=0.1,0.9

for J in 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0
do
  python $BIN --strengths $ALPHA --savedir $SDIR --buffer $BUFFER --length $LENGTH --coupling $J --savedir $SDIR --ntrials $N --strials $S --virtuals $V --layers $L
done

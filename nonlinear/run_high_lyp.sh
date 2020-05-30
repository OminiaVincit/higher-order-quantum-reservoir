#!/usr/bin/bash
BIN=source/lyapuv_highorder.py
Q=5
N=1
S=1e-3
L=5
SDIR=res_high_lyp_q$Q
BUFFER=4000
LENGTH=5000
V=1

ALPHA=0.1,0.9

export OMP_NUM_THREADS=3
for S in 1e-3 1e-4 1e-5
do

for J in 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0
do
  python $BIN --units $Q --strengths $ALPHA --savedir $SDIR --buffer $BUFFER --length $LENGTH --coupling $J --savedir $SDIR --ntrials $N --initial_distance $S --virtuals $V --layers $L
done
done

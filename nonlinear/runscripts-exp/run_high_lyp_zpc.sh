#!/usr/bin/bash
export OMP_NUM_THREADS=1

BIN=../source/lyapuv_hqrc.py
vals=$(seq 0.10 0.10 1.00)
Q=5
N=10
L=5
SDIR=../../../data/hqrc/lyp_$Q

BUFFER=10000
LENGTH=11000
V=1

for S in 1e-5 1e-10
do
for alpha in $vals
do
  python $BIN --units $Q --strength $alpha --savedir $SDIR --buffer $BUFFER --length $LENGTH --savedir $SDIR --ntrials $N --initial_distance $S --virtuals $V --layers $L
done
done

#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/eff_hqrc_tau_change.py
SAVE=../../../data/hqrc/eff_dim
vals=$(seq 0.00 0.01 1.00)
N=10

BUFFER=10000
LENGTH=11000

for V in 5 15
do
for SP in 1.0
do
for p in $vals
do
python $BIN --savedir $SAVE --buffer $BUFFER --length $LENGTH --sparsity $SP --strength $p --ntrials $N --virtuals $V
done
done
done
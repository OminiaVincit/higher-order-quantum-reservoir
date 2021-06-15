#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/eff_hqrc_tau_change.py
SAVE=../../../data/hqrc/eff_dim_nonlinear
vals=$(seq 0.00 0.05 1.00)
N=10

BUFFER=1000
LENGTH=1100

SM=1
SG=10.0

for V in 1
do
for SP in 1.0
do
for p in $vals
do
python $BIN --savedir $SAVE --buffer $BUFFER --length $LENGTH --sparsity $SP --nonlinear $SM --sigma_input $SG --strength $p --ntrials $N --virtuals $V
done
done
done
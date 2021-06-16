#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/eff_hqrc_tau_change.py
SAVE=../../../data/hqrc/eff_dim_mask_input
vals=$(seq 0.00 0.05 1.00)
N=10

BUFFER=10000
LENGTH=11000

SM=0
SG=1.0
MASK=1

for V in 1
do
for SP in 1.0
do
for p in $vals
do
python $BIN --mask_input $MASK --savedir $SAVE --buffer $BUFFER --length $LENGTH --sparsity $SP --nonlinear $SM --sigma_input $SG --strength $p --ntrials $N --virtuals $V
done
done
done
#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/eff_hqrc_tau_change.py
SAVE=../eff_dim
vals=$(seq 0.00 0.01 1.00)
N=10
V=1

for p in $vals
do
python $BIN --strength $p --ntrials $N --virtuals $V
done

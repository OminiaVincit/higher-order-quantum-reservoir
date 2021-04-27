#!/usr/bin/bash
# Script to calculate quantum esp
export OMP_NUM_THREADS=1

BIN=../source/hqrc_spectral.py
SAVE=../../../data/hqrc/delete

SPINS=5
TMIN=-7.0
TMAX=7.0
NTAUS=281
NPROCS=1

python $BIN --nspins $SPINS --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --nproc $NPROCS --savedir $SAVE

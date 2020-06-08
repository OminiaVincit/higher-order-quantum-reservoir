#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states.py
LENGTH=2000
BG=1000
ED=2000
SAVE=../test_states
QR=5
ALPHA=0.5
PROC=24

python $BIN --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --strength $ALPHA --nproc $PROC
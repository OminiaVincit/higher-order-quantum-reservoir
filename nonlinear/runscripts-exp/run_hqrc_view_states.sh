#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states.py
LENGTH=10000
BG=9000
ED=10000
SAVE=../rs_dynamics
QR=1
ALPHA=0.0
PROC=100
CONST=0
BASE=qrc_varj

for J in 1.0
do
python $BIN --basename $BASE --const $CONST --coupling $J --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --strength $ALPHA --nproc $PROC
done
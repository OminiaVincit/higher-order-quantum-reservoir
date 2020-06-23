#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/esp_hqrc.py
N=2
J=1.0
SAVE=../test_esp
QR=5
TAUS=\'-1,0,3\' # The parameters for tau is 2**x for x in TAUS
STRENGTHS='0.1,0.5,0.9'
VS='1,5'
QRS='5'

python $BIN --coupling $J --taudeltas $TAUS --layers $QRS --strengths $STRENGTHS --virtuals $VS --ntrials $N --savedir $SAVE

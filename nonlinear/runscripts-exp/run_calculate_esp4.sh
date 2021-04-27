#!/usr/bin/bash
# Script to calculate quantum esp
export OMP_NUM_THREADS=1

BIN=../source/esp_hqrc.py
S=10
N=10
J=1.0
QR=5
E=1000
SAVE=../../../data/hqrc/qesp_E_$E

for a in 0.0 0.5 0.9
do
for T in 100 200 500 1000 5000 10000
do
python $BIN --evalstep $E --strengths $a --layers $QR --buffer $T --coupling $J --strials $S --ntrials $N --savedir $SAVE
done
done

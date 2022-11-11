#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/lorentz_hqrc.py
UNITS=6
NQRC=6
SAVE=../../../data/hqrc/lorentz_QR_$NQRC\_$UNITS\_v202208

SOLVER=linear_pinv
DYNAMIC=phase_trans
TAU=10.0
DT=0.001
BUF=20
TRAIN=100
VAL=100

TYOP='X'
SIG=1.0
NON=0
LOAD=0
N=1
CB=1
STRENGTH='0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0'

for TYIN in 9
do
for V in 10
do
python $EXE --type_op $TYOP --tau $TAU --Ntrials $N --strengths $STRENGTH --load_result $LOAD --savedir $SAVE --units $UNITS --virtuals $V --nqrc $NQRC --nonlinear $NON --combine_input $CB --type_input $TYIN --solver $SOLVER --dynamic $DYNAMIC --dt $DT --T_buf $BUF --T_train $TRAIN --T_val $VAL
done
done
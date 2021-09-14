#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/lorentz_hqrc.py
SAVE=../../../data/hqrc/lorentz

UNITS=5
SOLVER=linear_pinv
DYNAMIC=phase_trans
TAU=10.0
DT=0.001
BUF=20
TRAIN=100
VAL=100

NQRC=6
CB=0
TYIN=0
SIG=1.0
NON=0
LOAD=0
N=1

for V in 1
do
python $EXE --tau $TAU --Ntrials $N --load_result $LOAD --savedir $SAVE --units $UNITS --virtuals $V --nqrc $NQRC --nonlinear $NON --combine_input $CB --type_input $TYIN --solver $SOLVER --dynamic $DYNAMIC --dt $DT --T_buf $BUF --T_train $TRAIN --T_val $VAL
done

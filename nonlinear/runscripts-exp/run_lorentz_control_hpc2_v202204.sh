#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/lorentz_hqrc_control.py
SAVE=../../../data/hqrc/lorentz_control

UNITS=5
SOLVER=linear_pinv
DYNAMIC=phase_trans
TAU=10.0
DT=0.001
BUF=10
TRAIN=10
VAL=10

NQRC=3
CB=1
TYIN=0
SIG=1.0
NON=0
LOAD=0
N=1
LORHO='28.0'
STRENGTH=0.0

for V in 10
do
python $EXE --strength $STRENGTH --lo_rhos $LORHO --tau $TAU --Ntrials $N --load_result $LOAD --savedir $SAVE --units $UNITS --virtuals $V --nqrc $NQRC --nonlinear $NON --combine_input $CB --type_input $TYIN --solver $SOLVER --dynamic $DYNAMIC --dt $DT --T_buf $BUF --T_train $TRAIN --T_val $VAL
done

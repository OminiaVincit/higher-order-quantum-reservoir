#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/lorentz_hqrc_control.py
SAVE=../../../data/hqrc/lorentz_control4

UNITS=5
SOLVER=linear_pinv
DYNAMIC=phase_trans
TAU=10.0
DT=0.001
BUF=20
TRAIN=100
VAL=100

CB=1
TYIN=0
SIG=1.0
NON=0
LOAD=0
N=1
LORHO='23.0,24.0'
STRENGTH=-1.0
PERTUBEDS='0.0,0.1'
MAINRHO=24.0

for NQRC in 3 6
do
for V in 5 10
do
python $EXE --main_rho $MAINRHO --controls $PERTUBEDS --strength $STRENGTH --lo_rhos $LORHO --tau $TAU --Ntrials $N --load_result $LOAD --savedir $SAVE --units $UNITS --virtuals $V --nqrc $NQRC --nonlinear $NON --combine_input $CB --type_input $TYIN --solver $SOLVER --dynamic $DYNAMIC --dt $DT --T_buf $BUF --T_train $TRAIN --T_val $VAL
done
done
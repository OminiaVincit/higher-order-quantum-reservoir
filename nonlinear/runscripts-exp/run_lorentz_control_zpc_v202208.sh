#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/lorentz_hqrc_control.py
SAVE=../../../data/hqrc/del_lorentz_control_v202208

UNITS=6
SOLVER=linear_pinv
DYNAMIC=phase_trans
TAU=10.0
DT=0.001
BUF=20
TRAIN=100
VAL=100

NQRC=6
CB=1

TYOP='X'
SIG=1.0
NON=0
LOAD=0
N=1
LORHO='22.0,23.0'
PERTUBEDS='2.0,1.0'
MAINRHO=28.0
STRENGTH='0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0'
TESTSTREN='0.01'

for TYIN in 5
do
for V in 10
do
python $EXE --test_strengths $TESTSTREN --type_op $TYOP --main_rho $MAINRHO --pertubed_strengths $PERTUBEDS --strengths $STRENGTH --lo_rhos $LORHO --tau $TAU --Ntrials $N --load_result $LOAD --savedir $SAVE --units $UNITS --virtuals $V --nqrc $NQRC --nonlinear $NON --combine_input $CB --type_input $TYIN --solver $SOLVER --dynamic $DYNAMIC --dt $DT --T_buf $BUF --T_train $TRAIN --T_val $VAL
done
done
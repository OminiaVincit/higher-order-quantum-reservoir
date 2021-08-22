#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runIPC.py
BINPLOT=../postprocess/plot_IPC_thres.py
PARENT=/data/zoro/hqrc
#T=2000000
T=2000000
TS=_T_$T
WD=50
VAR=4
DEG=4
#DELAYS='0,100,50,50,20,20,10,10'
#DELAYS='0,20,20,20,20,20,10,10'
DELAYS='0,100,50,50,20'
#
V='1'
TAUS='8.0'
NSPINS=5
NPROC=101
QR=5

THRES=0.0
DYNAMIC='full_random'
CAPA=25
WIDTH=0.01

AMIN=0.0
AMAX=1.0
NAS=100
EXP=0
CB=2

for MASK in 0
    do
    FRS='XXX'
    for SEED in 0 1 2 3 4
    do
    LBS=IPC_one_input_cb_$CB\_mask_$MASK\_seed_$SEED
    SAVE=$PARENT\/$LBS
    FRS=$FRS,$LBS

    python $EXE --combine_input $CB --mask_input $MASK --exp $EXP --amin $AMIN --amax $AMAX --nas $NAS --nqrc $QR --nproc $NPROC --spins $NSPINS --seed $SEED --dynamic $DYNAMIC --deg_delays $DELAYS --thres $THRES --virtuals $V --length $T --max_deg $DEG --max_window $WD --max_num_var $VAR --savedir $SAVE
    done

    P=mdeg_4_mvar_4

    for THRES in 0.0 1e-4
    do
    python $BINPLOT --solver 'linear_pinv_' --exp $EXP --parent $PARENT --folders $FRS --T $T --thres $THRES --amin $AMIN --amax $AMAX --nas $NAS --nqrc $QR --dynamic $DYNAMIC --virtuals $V --taus $TAUS --nspins $NSPINS --keystr $P  --max_capa $CAPA --width $WIDTH
    done
done


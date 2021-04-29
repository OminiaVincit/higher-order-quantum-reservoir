#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runIPC.py
BINPLOT=../postprocess/plot_IPC_thres.py
SAVE=../../../data/hqrc/IPC_seed2

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
SEED=2
CAPA=25
WIDTH=0.01
AMIN=0.9
AMAX=1.0
NAS=100
P=seed_2_mdeg_4_mvar_4_thres_0.0_delays_$DELAYS$TS


python $EXE --amin $AMIN --amax $AMAX --nas $NAS --nqrc $QR --nproc $NPROC --spins $NSPINS --seed $SEED --dynamic $DYNAMIC --deg_delays $DELAYS --thres $THRES --virtuals $V --length $T --max_deg $DEG --max_window $WD --max_num_var $VAR --savedir $SAVE

for THRES in 0.0 1e-4
do
python $BINPLOT --thres $THRES --amin $AMIN --amax $AMAX --nas $NAS --nqrc $QR --dynamic $DYNAMIC --virtuals $V --taus $TAUS --nspins $NSPINS --folder $SAVE --posfix $P  --max_capa $CAPA --width $WIDTH
done



#!/usr/bin/bash
for N in 1 2 3 4 5
do
python source/narma_nmse_multiplex.py --ntrials 10 --nqrc $N 
done

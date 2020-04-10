#!/usr/bin/bash
for a in 0.1 0.3 0.5 0.7 0.9
do
for N in 1 2 3 4 5 
  do
    python source/narma_nmse_deep.py --strength $a --ntrials 10 --nqrc $N 
done
done

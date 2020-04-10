#!/usr/bin/bash
for a in 1.0
  do
    python source/narma_nmse_highorder.py --strength $a --ntrials 10 
done

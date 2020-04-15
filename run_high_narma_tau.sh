#!/usr/bin/bash
for a in 0.0 0.1 0.3 0.5 0.7 0.9
  do
    python source/narma_nmse_highorder_tau_change.py --strength $a --ntrials 10 --layers 1,2,3,4,5
    python source/narma_nmse_highorder_V_change.py --strength $a --ntrials 10 --layers 1,2,3,4,5
done

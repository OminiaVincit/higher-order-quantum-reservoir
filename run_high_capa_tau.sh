#!/usr/bin/bash
for a in 0.0 0.1 0.3 0.5 0.7 0.9
  do
    python source/mem_capacity_highorder_tau_change.py --strength $a --ntrials 10 --maxd 256
done

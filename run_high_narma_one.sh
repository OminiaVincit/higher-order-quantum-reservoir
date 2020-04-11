#!/usr/bin/bash
for a in 0.0 1.0
  do
    python source/narma_nmse_highorder_V_oneinput.py --strength $a --ntrials 2 
done

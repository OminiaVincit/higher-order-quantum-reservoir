#!/usr/bin/bash
for o in 20 15 10 5 2
  do
    for a in 0.0
      do
        python source/narma_nmse_highorder_tau_change.py --layers 1 --units 5 --orders $o --strength $a --ntrials 10
        python source/narma_nmse_highorder_V_change.py --layers 1 --units 5 --orders $o --strength $a --ntrials 10
    done
  done

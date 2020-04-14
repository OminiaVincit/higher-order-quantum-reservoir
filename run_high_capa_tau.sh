#!/usr/bin/bash
for a in 0.3 0.7
  do
    python source/mem_capacity_highorder_tau_change.py --taskname qrc_stm --layers 2,3,4,5 --strength $a --ntrials 10 --nproc 51 --maxd 200
done

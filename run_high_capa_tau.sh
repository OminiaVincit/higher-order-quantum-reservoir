#!/usr/bin/bash
python source/mem_capacity_highorder_tau_change.py --taskname qrc_stm --layers 3,4,5 --strength 0.9 --ntrials 10 --nproc 51 --maxd 200
python source/mem_capacity_highorder_tau_change.py --taskname qrc_stm --layers 2,3,4,5 --strength 0.5 --ntrials 10 --nproc 51 --maxd 200

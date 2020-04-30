#!/usr/bin/bash

for a in 5 15 20
do
  python source/narma_nmse_highorder_tau_change.py --orders $a --savedir narma_deep_tau --deep 1 --ntrials 10 --virtuals 15 --layers 2,3,4,5 --strength 0.5
  python source/narma_nmse_highorder_tau_change.py --orders $a --savedir narma_deep_tau --deep 1 --ntrials 10 --virtuals 15 --layers 2,3,4,5 --strength 0.9
  python source/narma_nmse_highorder_strength_change.py --orders $a --savedir narma_deep_strength --deep 1 --ntrials 10 --virtuals 20 --taudelta 2.0 --layers 2,3,4,5 --strengths 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
done

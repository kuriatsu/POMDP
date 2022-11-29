#!/bin/bash

python3 ras_value_iteration_sweep.py param_14.yaml &
python3 ras_value_iteration_sweep.py param_14_2.yaml &
python3 ras_value_iteration_sweep.py param_14_3.yaml &
wait
python3 ras_value_iteration_sweep.py param_15.yaml &
python3 ras_value_iteration_sweep.py param_15_2.yaml &
python3 ras_value_iteration_sweep.py param_15_3.yaml &
wait
python3 ras_value_iteration_sweep.py param_16.yaml &
python3 ras_value_iteration_sweep.py param_16_2.yaml &
python3 ras_value_iteration_sweep.py param_16_3.yaml &
wait
python3 ras_value_iteration_sweep.py param_17.yaml &
python3 ras_value_iteration_sweep.py param_17_2.yaml &
wait
python3 ras_value_iteration_sweep.py param_18.yaml &
python3 ras_value_iteration_sweep.py param_18_2.yaml &

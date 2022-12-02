#!/bin/bash

python3 ras_value_iteration_sweep.py param_12_1.yaml &
python3 ras_value_iteration_sweep.py param_12_2.yaml &
python3 ras_value_iteration_sweep.py param_12_3.yaml &
python3 ras_value_iteration_sweep.py param_12_4.yaml &
python3 ras_value_iteration_sweep.py param_12_5.yaml &
wait
python3 ras_value_iteration_sweep.py param_10_1.yaml &
python3 ras_value_iteration_sweep.py param_10_2.yaml &
python3 ras_value_iteration_sweep.py param_10_3.yaml &
python3 ras_value_iteration_sweep.py param_10_4.yaml &
wait
python3 ras_value_iteration_sweep.py param_11_1.yaml &
python3 ras_value_iteration_sweep.py param_11_2.yaml &

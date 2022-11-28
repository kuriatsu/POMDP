#!/bin/bash

python3 ras_value_iteration_sweep.py param_2.yaml &
python3 ras_value_iteration_sweep.py param_3.yaml &
python3 ras_value_iteration_sweep.py param_4.yaml &
python3 ras_value_iteration_sweep.py param_5.yaml &
wait
python3 ras_value_iteration_sweep.py param_6.yaml &
python3 ras_value_iteration_sweep.py param_7.yaml &
python3 ras_value_iteration_sweep.py param_8.yaml &
python3 ras_value_iteration_sweep.py param_9.yaml &
wait
python3 ras_value_iteration_sweep.py param_10.yaml &
python3 ras_value_iteration_sweep.py param_11.yaml &
python3 ras_value_iteration_sweep.py param_12.yaml &
python3 ras_value_iteration_sweep.py param_13.yaml &

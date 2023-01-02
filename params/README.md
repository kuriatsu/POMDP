# params

delta_t: update time 

prediction_horizon:  

safety_margin: ego vehicle stop x[m] before the target 

ideal_speed: ego vehicle try to keep this speed

min_speed: pass through with this speed

ordinary_G : for keep speed

p_efficiency: panalty of acceleration/deceleration in front of the ambiguity  
```python
# L158
p_efficiency = acceleration_rate**2 * closest_amb
```

p_comfort: not used 

p_ambiguity: penalty when passed the ambiguity without deceleration 

p_bad_int_request: not used 

p_int_request: penalty when request intervention 

p_delta_t: not used 

goal_value: not used


operator_int_prob: operator decision prob (target has no risk) 

min_time: minimum time for intervention 

min_time_var: minimum time variance 

acc_time_min: accuracy when minimum time

acc_time_var: time-acc variance 

acc_time_slope: time-acc slope 

risk_positions: risk position list 



import numpy as np
forget_rate = 0.2
num_gradual = 10
n_epoch = 200
exponent = 1

rate_schedule = np.ones(n_epoch)*forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, 10)

print('rate schedule: ',rate_schedule)
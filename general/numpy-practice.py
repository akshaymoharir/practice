

# This file has general practice code statements for practicing numpy library.

import numpy as np
print("numpy version is:",np.__version__)

# Create numpy vectors
a = np.zeros(4)
b = np.ones(5)
print(f"vector a:{a}, shape of vector a:{a.shape}")
print(f"vector b:{b}, shape of vector b:{b.shape}")

# np.arange
# Parameters:
#       start: starting number integer or real. This number will be included in the arragenemnt formed.
#               This argument is optional, if not provided, np will take 0 as starting point.
#       stop: stop number integer or real. This number will be excluded in the arragement formed.
#               This argument is required, without this argument np wont know where to stop.
#       step: step for arrangement asked by user. This number will be used to form equidistant arrangement.
#               If step is specified, then starting point also needs to be provided. See examples below.

# Following statement will form vector
stop_only_vec = np.arange(5)
print(f"This vector specified by only stop argument to np_arange:{stop_only_vec}, shape:{stop_only_vec.shape}")

# Following statement will form vector with given a starting and stopping point
start_stop_vec = np.arange(5,13)
print(f"This vector has given start and stop point and will print all integers in between. vec:{start_stop_vec},"
            f" shape:{start_stop_vec.shape}")

# Simple array
simple_array = np.array([1,5,6,8])
print(f"simple array:{simple_array}, and its shape{simple_array.shape}")

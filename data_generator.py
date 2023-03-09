import numpy as np
import matplotlib.pyplot as plt
import SIR
import time



if __name__ == '__main__':
    t1 = time.time()
    coefficients = [.912,.216,.035]
    initial = np.array([1e6,1,0,0])
    solve = generate_SIRD(coefficients,initial,50)
    t2 = time.time()
    print(t2-t1)
    dates = np.arange(50)
    plt.plot(dates,solve[0,:],'s',label = "S")
    plt.plot(dates,solve[1,:],'s',label = "I")
    plt.plot(dates,solve[2,:],'s', label = 'R')
    plt.plot(dates,solve[3,:],'s', label = 'D')
    plt.legend()
    plt.show()

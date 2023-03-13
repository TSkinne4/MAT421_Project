import numpy as np
import matplotlib.pyplot as plt
import SIR
import time
import coef_find as cf


if __name__ == '__main__':
    days = 70
    dim = 100
    db = 0.4/dim
    dl = 0.4/dim
    norms = np.zeros((dim,dim),dtype = 'float64')
    t1 = time.time()
    coefficients = [0.6,.216,.035]
    initial = np.array([1e6,1,0,0])
    solve = SIR.generate_SIRD_curves(SIR.SIRD,coefficients,initial,days)

    for i in range(dim):
        for j in range(dim):
            test_coffs = [0.6,dl*(j+1),db*(i+1)]
            test_curve = SIR.generate_SIRD_curves(SIR.SIRD,test_coffs,initial,days)
            norms[i,j] = cf.difference(solve,test_curve,3,days)
            if ((j+1) == dim) and ((i+1)%(dim/10) == 0):
                percent = (i+1)/(dim)*100
                print(f'{percent}%')
    plt.imshow(norms, cmap='turbo', interpolation='nearest')
    plt.colorbar()

    t2 = time.time()
    print(t2-t1)
    #dates = np.arange(days)
    #plt.plot(dates,solve[0,:],'s',label = "S")
    #plt.plot(dates,solve[1,:],'s',label = "I")
    #plt.plot(dates,solve[2,:],'s', label = 'R')
    #plt.plot(dates,solve[3,:],'s', label = 'D')
    #plt.legend()
    plt.show()

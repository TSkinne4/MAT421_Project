import numpy as np
import matplotlib.pyplot as plt
import SIR
import time



if __name__ == '__main__':
    t1 = time.time()
    #Coeffients = [a,B,sigma,eta,mu,xi,tau,gamma,delta,lambda_0,lambda_1,k_0,k_1]
    coefficients = [0.012,0.7,0.06,0.01,0.1,0.1,0.5,0.72,0.8,0.2,0.01,0.035,0.02]
    initial = np.array([3e6,1.0,0.0,0.0,0.0,1.0,0.0,1.0])
    solve = SIR.generate_test_data(SIR.modfified_SIRD,coefficients,initial,100,100)
    t2 = time.time()
    print(t2-t1)
    dates = np.arange(100)
    #[S,I,R,D,C,E,A,Q]
    plt.plot(dates,solve[0,:],'s',label = "S")
    plt.plot(dates,solve[1,:],'s',label = "I")
    plt.plot(dates,solve[2,:],'s', label = 'R')
    plt.plot(dates,solve[3,:],'s', label = 'D')
    plt.plot(dates,solve[4,:],'s',label = "C")
    plt.plot(dates,solve[5,:],'s',label = "E")
    plt.plot(dates,solve[6,:],'s', label = 'A')
    plt.plot(dates,solve[7,:],'s', label = 'Q')
    plt.legend()
    plt.show()

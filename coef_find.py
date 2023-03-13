import matplotlib.pyplot as plt
import numpy as np
import SIR
import time

def difference(data,model,parameter_num,points):
    diff = np.sum(np.power(model[0:4,:]-data[0:4,:],2))
    error = np.sqrt(np.abs(diff)/np.sqrt(points-parameter_num))
    return error

def find_derivative(func,initial,coefficients,data,days,growth):
    solution = np.zeros_like(coefficients)
    for i in range(len(coefficients)):
        lower = coefficients[:]
        lower[i] = lower[i]-growth
        low_model = SIR.generate_SIRD_curves(func,lower,initial,days)
        low_diff = difference(data,low_model,len(coefficients),days)
        higher = coefficients[:]
        higher[i] = higher[i]+growth
        high_model = SIR.generate_SIRD_curves(func,higher,initial,days)
        high_diff = difference(data,high_model,len(coefficients),days)
        solution[i] = (high_diff-low_diff)/(2*growth)
    return solution

def find_min(func,data,guess,initial,iterations,tol,days,growth):
    current = guess
    i = 0
    current_error = difference(data,SIR.generate_SIRD_curves(func,current,initial,days),len(guess),days)
    while i <= iterations:
        if i%50 == 0:
            current_error = difference(data,SIR.generate_SIRD_curves(func,current,initial,days),len(guess),days)
            print(f'{i}\t {current_error}')
        current = current - growth*find_derivative(func,initial,current,data,days,growth)/1e3
        i += 1
    return current

if __name__ == '__main__':
    t1 = time.time()
    coefficients = np.array([.912,.216,.035])
    initial = np.array([1e6,1,0,0])
    days = 100
    data = SIR.generate_test_data(SIR.SIRD,coefficients,initial,days,100)
    guess = np.array([1,.5,0.01])
    guess = find_min(SIR.SIRD,data,guess,initial,1000,1e-2,days,1e-6)
    print(f'guess:{guess}')
    t2 = time.time()

    solve = SIR.generate_test_data(SIR.SIRD,guess,initial,100,100)
    dates = np.arange(100)

    plt.plot(dates,solve[0,:])
    plt.plot(dates,data[0,:],'s')
    #plt.plot(dates,solve[1,:],'s',label = "I")
    #plt.plot(dates,solve[2,:],'s', label = 'R')
    #plt.plot(dates,solve[3,:],'s', label = 'D')
    plt.show()
    print(t2-t1)

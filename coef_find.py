import matplotlib.pyplot as plt
import numpy as np
import SIR
import time

def difference(data,model,parameter_num,points):
    '''Finds the error between the data an the model, Comparing only S,I,R, and D'''
    #diff = np.sum(np.power(model[0:4,:]-data[0:4,:],2))
    diff = np.sum(np.power(model[:,:]-data[:,:],2))
    error = np.sqrt(np.abs(diff)/np.sqrt(points-parameter_num))
    return error

def find_derivative(func,initial,coefficients,data,days,growth):
    '''Computes the how the function changes with changes when varying the coefficients
    Esentially a gradient'''
    solution = np.zeros_like(coefficients)
    for i in range(len(coefficients)): #Finds derivative with respect to each coefficient
        #Finds the error of the function for a smaller coefficient
        lower = coefficients[:]
        lower[i] = lower[i]-growth
        low_model = SIR.generate_SIRD_curves(func,lower,initial,days)
        low_diff = difference(data,low_model,len(coefficients),days)
        #Finds the error of the function for a larger coefficient
        higher = coefficients[:]
        higher[i] = higher[i]+growth
        high_model = SIR.generate_SIRD_curves(func,higher,initial,days)
        high_diff = difference(data,high_model,len(coefficients),days)
        #Central Difference
        solution[i] = (high_diff-low_diff)/(2*growth)
    return solution

def find_min(func,data,guess,initial,iterations,tol,days,growth):
    '''Determines the coefficients which minimize the difference between the data and the model'''
    current = guess
    #Initial so that we can get an x_{n-1} term
    current_error = difference(data,SIR.generate_SIRD_curves(func,current,initial,days),len(guess),days)
    der = find_derivative(func,initial,current,data,days,growth)
    past = current[:]
    current = current - growth*der/1e6
    i = 1

    while i <= iterations:
        past_der = der[:] #Stores previous derivative and updates
        der = find_derivative(func,initial,current,data,days,growth)

        current_error = difference(data,SIR.generate_SIRD_curves(func,current,initial,days),len(guess),days)
        if current_error < tol: #Runs until our error is less than this value
            print(f'{i}\t {current_error}')
            break
        if i%100 == 0:
            print(f'Iteration:{i}\tCurrent error: {current_error}')
            #print(current)
        if (np.sum(np.power(der-past_der,2))) == 0:
            print("Aborted due to too small a change")
            break
        growth = np.abs(np.dot(current-past,der-past_der))/(np.sum(np.power(der-past_der,2))) #updating growth term

        past = current[:] #Stores previous value and updates
        current = current - growth*der
        current[current<0] = 1e-6
        i += 1
    return current

if __name__ == '__main__':

    #Menu of options

    '''t1 = time.time()
    coefficients = np.array([.912,.216,.035])
    initial = np.array([1e6,1,0,0])
    days = 100
    data = SIR.generate_test_data(SIR.SIRD,coefficients,initial,days,100)
    guess = np.array([0.5,0.5,0.5])
    pop = np.sum(initial)
    #guess[0] = -(data[0,2]-data[0,0])/2*pop/(data[0,1]*data[1,1])
    #guess[1] = (data[2,2]-data[2,0])/(2*data[1,1])
    #guess[2] = (data[3,2]-data[3,0])/(2*data[1,1])
    guess = find_min(SIR.SIRD,data,guess,initial,1e3,1e2,days,1e-6)
    print(f'guess:{guess}')
    t2 = time.time()

    solve = SIR.generate_test_data(SIR.SIRD,guess,initial,100,100)
    dates = np.arange(100)

    plt.plot(dates,solve[0,:])
    plt.plot(dates,data[0,:],'s')
    #plt.plot(dates,solve[1,:],'s',label = "I")
    #plt.plot(dates,solve[2,:],'s', label = 'R')
    #plt.plot(dates,solve[3,:],'s', label = 'D')
    plt.show()'''

    #Test for modified

    t1 = time.time()
    coefficients = [0.012,0.7,0.06,0.01,0.1,0.1,0.5,0.72,0.8,0.2,0.01,0.035,0.02]
    initial = np.array([3e6,1.0,0.0,0.0,0.0,1.0,0.0,1.0])
    days = 100
    data = SIR.generate_test_data(SIR.modified_SIRD,coefficients,initial,days,100)
    #guess = 0.5*np.ones(13,dtype = 'float64')
    guess = np.array([0.02,0.4,0.04,0.04,0.6,0.7,0.6,0.42,0.4,0.4,0.05,0.03,0.01])
    pop = np.sum(initial)
    #guess[0] = -(data[0,2]-data[0,0])/2*pop/(data[0,1]*data[1,1])
    #guess[1] = (data[2,2]-data[2,0])/(2*data[1,1])
    #guess[2] = (data[3,2]-data[3,0])/(2*data[1,1])
    guess = find_min(SIR.modified_SIRD,data,guess,initial,1e3,1e2,days,1e-6)
    print(f'guess:{guess}')
    t2 = time.time()

    solve = SIR.generate_test_data(SIR.modified_SIRD,guess,initial,100,100)
    dates = np.arange(100)

    plt.plot(dates,solve[0,:])
    plt.plot(dates,solve[1,:])
    plt.plot(dates,solve[2,:])
    plt.plot(dates,solve[3,:])
    #plt.plot(dates,data[0,:],'s')
    #plt.plot(dates,data[1,:],'s')
    #plt.plot(dates,data[2,:],'s')
    #plt.plot(dates,data[3,:],'s')
    #plt.plot(dates,solve[1,:],'s',label = "I")
    #plt.plot(dates,solve[2,:],'s', label = 'R')
    #plt.plot(dates,solve[3,:],'s', label = 'D')
    plt.show()

    print(t2-t1)

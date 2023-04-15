import matplotlib.pyplot as plt
import numpy as np
import SIR
import time

def difference(data,model,parameter_num,points):
    '''Finds the error between the data an the model, Comparing only S,I,R, and D'''
    diff = np.sum(np.power(model[1:4,:]-data[1:4,:],2))
    #diff = np.sum(np.power(model[1:4,:]-data[1:4,:],2))
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
    print(current)
    current = current/0.5*0.25
    print(current)
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
    print(f'Completed on iteration {i} with a final error of {current_error}')
    return current

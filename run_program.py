import SIR
import coef_find
import time
import numpy as np
import matplotlib.pyplot as plt
import coef_find as cf
from numpy import random

def display_SIRD(solve):
    days = len(solve[0,:])
    dates =range(days)

    plt.plot(dates,solve[0,:],'b',label = "S")
    plt.plot(dates,solve[1,:],'r',label = "I")
    plt.plot(dates,solve[2,:],'g', label = 'R')
    plt.plot(dates,solve[3,:],'k', label = 'D')

    plt.xlim(0,days)

    plt.xlabel('Time (days)')
    plt.ylabel('People')

    plt.grid()
    plt.legend()
    plt.show()

def display_fit(fit,data):
    days = len(fit[0,:])
    dates =range(days)
    gap = int(days/10)
    data_dates = np.arange(0,days,gap)

    plt.plot(dates,fit[0,:],'b',label = "S")
    plt.plot(dates,fit[1,:],'r',label = "I")
    plt.plot(dates,fit[2,:],'g', label = 'R')
    plt.plot(dates,fit[3,:],'k', label = 'D')

    plt.plot(data_dates,data[0,::gap],'bs')
    plt.plot(data_dates,data[1,::gap],'rs')
    plt.plot(data_dates,data[2,::gap],'gs')
    plt.plot(data_dates,data[3,::gap],'ks')

    plt.xlim(0,days)

    plt.xlabel('Time (days)')
    plt.ylabel('People')

    plt.grid()
    plt.legend()
    plt.show()

decision = int(input('Enter the number corresponding to the program you would like to run\n\
1) Generate example SIRD curves\n\
2) Generate example modified SIRD curves\n\
3) Fit SIRD to known\n\
4) Fit SIRD to random\n\
5) Fit modified SIRD\n\
6) SIRD min vizualizer\n\
7) Modified SIRD min visuazlizer\n\
8) Modified SIRD min discrepancy\n\
9) Compartmental Model\n'))

days = 250 #Number of days program will be ran for

t1 = time.time()
if decision == 1:
    #Model Coefficients
    coefficients = [0.2,0.1,0.01]
    initial = np.array([3e6,1.0,0.0,0.0])

    solve = SIR.generate_SIRD_curves(SIR.SIRD,coefficients,initial,days)
    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_SIRD(solve)

elif decision == 2:
    #Coeffients = [a,B,sigma,eta,mu,xi,tau,gamma,delta,lambda_0,lambda_1,k_0,k_1]
    coefficients = [0.05,0.5,0.1,0.05,0.01,0.33,0.5,0.5,0.25,0.05,0.01,0.035,0.02]
    initial = np.array([3e6,1.0,0.0,0.0,0.0,1.0,0.0,1.0])

    solve = SIR.generate_SIRD_curves(SIR.modified_SIRD,coefficients,initial,days)
    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_SIRD(solve)

elif decision == 3:
    coefficients = [0.2,0.1,0.01]
    initial = np.array([3e6,1.0,0.0,0.0])
    data = SIR.generate_SIRD_curves(SIR.SIRD,coefficients,initial,days)

    guess = np.array([0.5,0.5,0.5])
    tol = 1e2
    iterations = 1e5

    fit_param = cf.find_min(SIR.SIRD,data,guess,initial,iterations,tol,days,1e-4)
    print(f'Coefficients for data {coefficients}\nCoefficients from fit {fit_param}')


    fit = SIR.generate_SIRD_curves(SIR.SIRD,fit_param,initial,days)

    final_error = cf.difference(data,fit,len(coefficients),len(fit[0,:]))
    print(f'Final Error: {final_error}')

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_fit(fit,data)
elif decision == 4:
    coefficients = np.random.rand(1,3)[0]
    coefficients[1] = coefficients[1]/10
    coefficients[2] = coefficients[2]/10

    initial = np.array([3e6,1.0,0.0,0.0])
    data = SIR.generate_SIRD_curves(SIR.SIRD,coefficients,initial,days)

    guess = np.array([0.5,0.5,0.5])
    tol = 1e3
    iterations = 1e4

    fit_param = cf.find_min(SIR.SIRD,data,guess,initial,iterations,tol,days,1e-4)
    print(f'Coefficients for data {coefficients}\nCoefficients from fit {fit_param}')


    fit = SIR.generate_SIRD_curves(SIR.SIRD,fit_param,initial,days)

    final_error = cf.difference(data,fit,len(coefficients),len(fit[0,:]))
    print(f'Final Error: {final_error}')

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_fit(fit,data)
elif decision == 5:
    coefficients = [0.012,0.7,0.06,0.01,0.1,0.1,0.5,0.72,0.8,0.2,0.01,0.035,0.02]
    initial = np.array([3e6,1.0,0.0,0.0,0.0,1.0,0.0,1.0])

    data = SIR.generate_SIRD_curves(SIR.modified_SIRD,coefficients,initial,days)

    guess = np.ones_like(coefficients)/2
    tol = 1e4
    iterations = 1e4

    fit_param = cf.find_min(SIR.modified_SIRD,data,guess,initial,iterations,tol,days,1e-4)
    print(f'Coefficients for data {coefficients}\nCoefficients from fit {fit_param}')


    fit = SIR.generate_SIRD_curves(SIR.modified_SIRD,fit_param,initial,days)

    final_error = cf.difference(data,fit,len(coefficients),len(fit[0,:]))
    print(f'Final Error: {final_error}')

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_fit(fit,data)

elif decision == 6:
    dim = 100
    db = 0.4/dim
    dl = 0.4/dim
    norms = np.zeros((dim,dim),dtype = 'float64')

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

    plt.xlabel('$\lambda$')
    plt.ylabel('$k_d$')

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    plt.show()



elif decision == 7:
    dim = 200
    db = 0.4/dim
    dl = 0.4/dim
    norms = np.zeros((dim,dim),dtype = 'float64')

    coefficients = [0.012,0.7,0.06,0.01,0.1,0.1,0.5,0.72,0.8,0.2,0.01,0.035,0.02]
    initial = np.array([1e6,1.0,0.0,0.0,0.0,1.0,0.0,1.0])
    solve = SIR.generate_SIRD_curves(SIR.modified_SIRD,coefficients,initial,days)

    for i in range(dim):
        for j in range(dim):
            test_coffs = [0.012,0.7,0.06,0.01,0.1,0.1,0.5,0.72,0.8,dl*(j+1),0.01,db*(i+1),0.02]
            test_curve = SIR.generate_SIRD_curves(SIR.modified_SIRD,test_coffs,initial,days)
            norms[i,j] = cf.difference(solve,test_curve,len(coefficients),days)
            if ((j+1) == dim) and ((i+1)%(dim/10) == 0):
                percent = (i+1)/(dim)*100
                print(f'{percent}%')
    plt.imshow(norms, cmap='turbo', interpolation='nearest')
    plt.colorbar()

    plt.xlabel('$\lambda$')
    plt.ylabel('$k_d$')

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    plt.show()

elif decision == 8:

    dim = 200
    db = 1/dim
    norms = np.zeros((dim),dtype = 'float64')

    coefficients = [0.012,0.7,0.06,0.01,0.1,0.1,0.5,0.72,0.8,0.2,0.01,0.035,0.02]
    initial = np.array([1e6,1.0,0.0,0.0,0.0,1.0,0.0,1.0])
    solve = SIR.generate_SIRD_curves(SIR.modified_SIRD,coefficients,initial,days)

    tested_coef = db*np.arange(1,dim+1)

    for i in range(dim):
        test_coffs = [0.012,0.7,0.06,0.01,0.1,0.05,0.5,db*(i+1),0.8,0.2,0.01,0.035,0.02]
        test_curve = SIR.generate_SIRD_curves(SIR.modified_SIRD,test_coffs,initial,days)
        norms[i] = cf.difference(solve,test_curve,len(coefficients),days)
        if (i+1)%(dim/10) == 0:
            percent = (i+1)/(dim)*100
            print(f'{percent}%')
    plt.plot(tested_coef,norms)


    plt.xlabel('$\lambda$')
    plt.ylabel('$k_d$')
    plt.grid()
    plt.xlim(0,1)

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    plt.show()

elif decision == 9:
    print('ERROR: Not Implemented')
    pass
else:
    print("ERROR: Input not recognized")


#[2.02064903e-02 3.86456809e-01 3.35613070e-02 1.00000000e-06
 #5.97331381e-01 6.99022582e-01 5.99751611e-01 4.19059990e-01
 #.13465992e-01 3.90305352e-01 2.71017126e-02 4.22264712e-02
 #1.99198264e-02] Discrepancy coefficients

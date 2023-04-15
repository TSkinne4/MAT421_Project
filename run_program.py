import SIR
import coef_find
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import coef_find as cf
from numpy import random

def display_SIRD(solve,labels = ['S','I','R','D']):
    days = len(solve[0,:])
    dates =range(days)

    plt.plot(dates,solve[0,:],'b',label = labels[0])
    plt.plot(dates,solve[1,:],'r',label = labels[1])
    plt.plot(dates,solve[2,:],'g', label = labels[2])
    plt.plot(dates,solve[3,:],'k', label = labels[3])

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
5) Fit modified SIRD to known\n\
6) SIRD min vizualizer\n\
7) Modified SIRD min visuazlizer\n\
8) Compartmental Model\n\
9) Peak infection comparison\n\
10) Extrapolation\n'))

days = 250 #Number of days program will be ran for

t1 = time.time()
if decision == 1:
    #Model Coefficients
    coefficients = [0.225,0.1,0.01]
    initial = np.array([3e6-1,1.0,0.0,0.0])

    solve = SIR.generate_SIRD_curves(SIR.SIRD,coefficients,initial,days)
    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_SIRD(solve)

elif decision == 2:
    #Coeffients = [a,B,sigma,eta,mu,xi,tau,gamma,delta,lambda_0,lambda_1,k_0,k_1]

    coefficients =  [0.005,0.225,0.225,0.01,0.02,0.33,0.25,0.25,0.125,0.1,2e-4,0.01,2e-5]
    initial = np.array([3e6-1,1.0,0.0,0.0,0.0,0.0,0.0,0.0])

    solve = SIR.generate_SIRD_curves(SIR.modified_SIRD,coefficients,initial,days)
    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_SIRD(solve)

    dates= np.arange(0,days)
    plt.plot(dates,solve[0,:],'b',label = 'S')
    plt.plot(dates,solve[1,:],'r',label = 'I')
    plt.plot(dates,solve[2,:],'g', label = 'R')
    plt.plot(dates,solve[3,:],'k', label = 'D')
    plt.plot(dates,solve[4,:],'c',label = 'C')
    plt.plot(dates,solve[5,:],'y',label = 'E')
    plt.plot(dates,solve[6,:],'m', label = 'A')
    plt.plot(dates,solve[7,:],color = mcolor.CSS4_COLORS['lime'], label = 'Q')
    plt.legend()
    plt.xlim(0,days)

    plt.xlabel('Time (days)')
    plt.ylabel('People')

    plt.grid()
    plt.show()

elif decision == 3:
    coefficients = [0.225,0.1,0.01]
    initial = np.array([3e6-1,1.0,0.0,0.0])
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

    initial = np.array([3e6-1,1.0,0.0,0.0])
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
    coefficients = [0.005,0.225,0.225,0.01,0.02,0.33,0.25,0.25,0.125,0.1,2e-4,0.01,2e-5]
    initial = np.array([3e6-1,1.0,0.0,0.0,0.0,0.0,0.0,0.0])

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
    db = 0.175/dim
    dl = 0.175/dim
    norms = np.zeros((dim,dim),dtype = 'float64')

    coefficients = [0.225,0.1,0.01]
    initial = np.array([3e6-1,1,0,0])
    solve = SIR.generate_SIRD_curves(SIR.SIRD,coefficients,initial,days)

    for i in range(dim):
        for j in range(dim):
            test_coffs = [0.225,dl*(j+1),db*(i+1)]
            test_curve = SIR.generate_SIRD_curves(SIR.SIRD,test_coffs,initial,days)
            norms[i,j] = cf.difference(solve,test_curve,3,days)
            if ((j+1) == dim) and ((i+1)%(dim/10) == 0):
                percent = (i+1)/(dim)*100
                print(f'{percent}%')
    plt.imshow(norms, cmap='turbo', interpolation='nearest', origin='lower', extent=[0, 0.2, 0, 0.2])
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

    coefficients = [0.005,0.225,0.225,0.01,0.02,0.33,0.25,0.25,0.125,0.1,2e-4,0.01,2e-5]
    initial = np.array([1e6-1,1.0,0.0,0.0,0.0,0,0.0,0])
    solve = SIR.generate_SIRD_curves(SIR.modified_SIRD,coefficients,initial,days)

    for i in range(dim):
        for j in range(dim):
            test_coffs = [0.005,0.225,0.225,0.01,0.02,0.33,0.25,0.25,0.125,dl*(j+1),2e-4,db*(i+1),2e-5]
            test_curve = SIR.generate_SIRD_curves(SIR.modified_SIRD,test_coffs,initial,days)
            norms[i,j] = cf.difference(solve,test_curve,len(coefficients),days)
            if ((j+1) == dim) and ((i+1)%(dim/10) == 0):
                percent = (i+1)/(dim)*100
                print(f'{percent}%')
    plt.imshow(norms, cmap='turbo', interpolation='nearest', origin='lower', extent=[0, 0.2, 0, 0.2])
    plt.colorbar()

    plt.xlabel('$\lambda$')
    plt.ylabel('$k_d$')

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    plt.show()

elif decision == 8:
    # [0.6,.216,.035]
    initial = np.array([1e6/4,1e6/4,1e6/4,1e6/4,1,1,1,1,0,0])
    coefficients = [0.225,0.1,0.01,0.5,0.75,0.5,0.25,0.5,0.25]
    #[B,lambda,k,c1,c2,d1,d2,f,g]
    #[0.6,0.216,0.035,0.5,0.75,0.25,0.5,0.5,0.25]

    solve = SIR.generate_SIRD_curves(SIR.compartmental_SIRD,coefficients,initial,days)
    sol = np.zeros_like(solve[:4,:])
    sol[0,:] = solve[0,:]+solve[1,:]+solve[2,:]+solve[3,:]
    sol[1,:] = solve[4,:]+solve[5,:]+solve[6,:]+solve[7,:]
    sol[2,:] = solve[8,:]
    sol[3,:] = solve[9,:]
    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_SIRD(sol)
    display_SIRD(solve,labels = ['Unvaccinated/No mask','Unvaccinate/Mask','Vaccinated/No mask','Vaccinated/Mask'])
    display_SIRD(solve[4:8],labels = ['Unvaccinated/No mask','Unvaccinate/Mask','Vaccinated/No mask','Vaccinated/Mask'])

elif decision == 9:
    # [0.6,.216,.035]

    coefficients = [0.225,0.1,0.01,0.5,0.75,0.5,0.25,0.5,0.25]
    #[B,lambda,k,c1,c2,d1,d2,f,g]
    #[0.6,0.216,0.035,0.5,0.75,0.25,0.5,0.5,0.25]
    total_s= 1e6
    dim = 100
    da = 1/dim
    db = 1/dim

    maxes = np.zeros((dim,dim))
    max_day = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(dim):
            initial = np.array([total_s*(1-da*i)*(1-j*db),total_s*da*i*(1-db*j),total_s*(1-da*i)*db*j,total_s*da*i*db*j,1,1,1,1,0,0])
            solve = SIR.generate_SIRD_curves(SIR.compartmental_SIRD,coefficients,initial,days)
            sol = np.zeros_like(solve[:4,:])
            sol[0,:] = solve[0,:]+solve[1,:]+solve[2,:]+solve[3,:]
            sol[1,:] = solve[4,:]+solve[5,:]+solve[6,:]+solve[7,:]
            sol[2,:] = solve[8,:]
            sol[3,:] = solve[9,:]
            maxes[i,j] = np.max(sol[1,:])
            #max_day[i,j] = np.argmax(sol[1,:])
            if ((j+1) == dim) and ((i+1)%(dim/10) == 0):
                percent = (i+1)/(dim)*100
                print(f'{percent}%')
    plt.imshow(maxes, cmap='turbo', interpolation='nearest')

    plt.colorbar()

    plt.xlabel('Percent Wearing Masks')
    plt.ylabel('Percent Vaccinated')

    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    plt.show()
    plt.imshow(max_day, cmap='turbo', interpolation='nearest')
    plt.colorbar()

    plt.xlabel('Percent Wearing Masks')
    plt.ylabel('Percent Vaccinated')
    plt.show()
elif  decision == 10:
    extrap_days = int(days/8)
    coefficients = [0.005,0.225,0.225,0.01,0.02,0.33,0.25,0.25,0.125,0.1,2e-4,0.01,2e-5]
    initial = np.array([1e6,1.0,0.0,0.0,0.0,1.0,0.0,1.0])

    tol = 1e4
    iterations = 1e4

    mod_guess = 0.5*np.ones_like(coefficients,dtype = 'float64')
    reg_guess =  np.array([0.5,0.5,0.5])
    reg_initial = np.array([1e6+2.0,1.0,0.0,0.0])

    full_data = SIR.generate_SIRD_curves(SIR.modified_SIRD,coefficients,initial,days)
    data = full_data[:,:extrap_days]
    reg_coffs = cf.find_min(SIR.SIRD,data[0:4,:],reg_guess,reg_initial,iterations,tol,extrap_days,1e-4)
    mod_coffs =  cf.find_min(SIR.modified_SIRD,data,mod_guess,initial,iterations,tol,extrap_days,1e-4)

    dates =range(days)
    gap = int(days/10)
    data_dates = np.arange(0,days,gap)

    solve_reg = SIR.generate_SIRD_curves(SIR.SIRD,reg_coffs,reg_initial,days)
    solve_mod = SIR.generate_SIRD_curves(SIR.modified_SIRD,mod_coffs,initial,days)

    t2 = time.time()
    total_time = t2-t1
    print(f'Simulation Elapsed in {total_time} seconds.')
    plt.plot(dates,solve_reg[1,:], label = 'Regular')
    plt.plot(dates,solve_mod[1,:], label = 'Modified')
    plt.plot(np.arange(0,extrap_days),data[1,:],'s',label = 'Supplied Data')
    plt.plot(np.arange(extrap_days,days,25),full_data[1,extrap_days:days:25],'s',label = 'Future Data')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('People')
    plt.xlim([0,days])
    plt.show()

else:
    print("ERROR: Input not recognized")


#[2.02064903e-02 3.86456809e-01 3.35613070e-02 1.00000000e-06
 #5.97331381e-01 6.99022582e-01 5.99751611e-01 4.19059990e-01
 #.13465992e-01 3.90305352e-01 2.71017126e-02 4.22264712e-02
 #1.99198264e-02] Discrepancy coefficients

import SIR
import coef_find
import time
import numpy as np
import matplotlib.pyplot as plt

def display_SIRD(solve):
    days = len(solve[0,:])
    dates =range(days)

    plt.plot(dates,solve[0,:],label = "S")
    plt.plot(dates,solve[1,:],label = "I")
    plt.plot(dates,solve[2,:], label = 'R')
    plt.plot(dates,solve[3,:], label = 'D')

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

    solve = SIR.generate_SIRD_curves(SIR.modfified_SIRD,coefficients,initial,days)
    t2 = time.time()
    total_time = t2-t1

    print(f'Simulation Elapsed in {total_time} seconds.')
    display_SIRD(solve)

elif decision == 3:
    print('ERROR: Not Implemented')
    pass
elif decision == 4:
    print('ERROR: Not Implemented')
    pass
elif decision == 5:
    print('ERROR: Not Implemented')
    pass
elif decision == 6:
    print('ERROR: Not Implemented')
    pass
elif decision == 7:
    print('ERROR: Not Implemented')
    pass
elif decision == 8:
    print('ERROR: Not Implemented')
    pass
elif decision == 9:
    print('ERROR: Not Implemented')
    pass
else:
    print("ERROR: Input not recognized")


#[2.02064903e-02 3.86456809e-01 3.35613070e-02 1.00000000e-06
 #5.97331381e-01 6.99022582e-01 5.99751611e-01 4.19059990e-01
 #.13465992e-01 3.90305352e-01 2.71017126e-02 4.22264712e-02
 #1.99198264e-02] Discrepancy coefficients

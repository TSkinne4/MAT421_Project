import numpy as np
import matplotlib.pyplot as plt
from numpy import random
def SIRD(partitions,coefficients,population,t):
    """Finds dX/dt, where X is a vector containing the partitions of the population"""
    d_part = np.zeros(4,dtype='float64')
    d_part[0] = -coefficients[0]*partitions[0]*partitions[1]/population #dS/dt
    d_part[1] = coefficients[0]*partitions[0]*partitions[1]/population-coefficients[1]*partitions[1]-coefficients[2]*partitions[1] #dI/dt
    d_part[2] = coefficients[1]*partitions[1] #dR/dt
    d_part[3] = coefficients[2]*partitions[1] #dD/dt
    return d_part

def modified_SIRD(part,coef,population,t):
    """Coeffients = [a,B,sigma,eta,mu,xi,tau,gamma,delta,lambda_0,lambda_1,k_0,k_1]
        X = [S,I,R,D,C,E,A,Q]
    """
    a,B,sigma,eta,mu,xi,tau,gamma,delta,lambda_0,lambda_1,k_0,k_1 = coef[0],coef[1],coef[2],coef[3],coef[4],coef[5],coef[6],coef[7],coef[8],coef[9],coef[10],coef[11],coef[12]
    S,I,R,D,C,E,A,Q = part[0],part[1],part[2],part[3],part[4],part[5],part[6],part[7]
    d_part = np.zeros(8,dtype = 'float64')
    d_part[0] = -a*S-B*I*S/population-sigma*S*A/population-eta*S
    d_part[1] = tau*A+gamma*E-delta*I
    d_part[2] = lambda_0*np.exp(-lambda_1*t)*Q
    d_part[3] = k_0*np.exp(-k_1*t)*Q
    d_part[4] = a*S-mu*C
    d_part[5] = -gamma*E+B*I*S/population+mu*C+eta*S+sigma*S*A/population-xi*E
    d_part[6] = -tau*A+xi*E
    d_part[7] = -lambda_0*np.exp(-lambda_1*t)*Q-k_0*np.exp(-k_1*t)*Q+delta*I
    return d_part


def generate_SIRD_curves(func,coefficients,initial,days):
    """Generates an SIRD curve from initial conditions"""
    population = np.sum(initial)
    curves = np.zeros((len(initial),days),dtype='float64')
    curves[:,0] = initial
    for i in range(1,days): #RK4
        k1 = func(curves[:,i-1],coefficients,population,i)
        k2 = func(curves[:,i-1]+k1/2,coefficients,population,i+1/2)
        k3 = func(curves[:,i-1]+k2/2,coefficients,population,i+1/2)
        k4 = func(curves[:,i-1]+k3,coefficients,population,i+1)

        curves[:,i] = curves[:,i-1]+(k1+2*k2+2*k3+k4)/6
        #for j in range(len(curves[:,i])):
        #    if curves[j,i] < 0:
        #        curves[j,i] = 0
        #print(f'c{curves[:,i]}')
        error = (np.sum(curves[:,i])/population-1)*100
        difference = np.sum(curves[:,i])-population
        #print(f'{difference}\t{error}')

    return curves

def generate_test_data(func,coefficients,initial,days,strength):
    data = generate_SIRD_curves(func,coefficients,initial,days)
    #data += random.randint(np.max(data)/strength,size = np.shape(data))
    return data

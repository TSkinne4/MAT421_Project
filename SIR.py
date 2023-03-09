import numpy as np
import matplotlib.pyplot as plt

def SIRD(partitions,coefficients,population):
    """Finds dX/dt, where X is a vector containing the partitions of the population"""
    d_part = np.zeros(4,dtype='float64')
    d_part[0] = -coefficients[0]*partitions[0]*partitions[1]/population #dS/dt
    d_part[1] = coefficients[0]*partitions[0]*partitions[1]/population-coefficients[1]*partitions[1]-coefficients[2]*partitions[1] #dI/dt
    d_part[2] = coefficients[1]*partitions[1] #dR/dt
    d_part[3] = coefficients[2]*partitions[1] #dD/dt
    return d_part

def generate_SIRD_curves(coefficients,initial,days):
    """Generates an SIRD curve from initial conditions"""
    population = np.sum(initial)
    curves = np.zeros((len(initial),days),dtype='float64')
    curves[:,0] = initial
    for i in range(1,days): #RK4
        k1 = SIRD(curves[:,i-1],coefficients,population)
        k2 = SIRD(curves[:,i-1]+k1/2,coefficients,population)
        k3 = SIRD(curves[:,i-1]+k2/2,coefficients,population)
        k4 = SIRD(curves[:,i-1]+k3,coefficients,population)
        curves[:,i] = curves[:,i-1]+(k1+2*k2+2*k3+k4)/6
    return curves

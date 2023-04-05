 #This file takes the data from a file that contains the total infected, the dead, and the recovered and then returns it as a numpy array
#Note that the data is cut as at some point, the recovered data was no longer kept track of
import csv
import numpy as np
import matplotlib.pyplot as plt


def extract_data_from_file(file_name):
    with open(file_name,'r') as file:
        reader = csv.reader(file)
        num = 0
        data = np.zeros((2,2))
        for row in reader:
            if num == 0:
                data = np.zeros((4,len(row)-4),dtype = 'float64')
                data[0,:] = np.arange(0,len(row)-4)
                num  += 1
                continue
            data[num,:] = np.array(row)[4:]
            num  += 1
    return data

def extract_data(file_name,days_infected,date_range):
    data = extract_data_from_file(file_name)
    data = data[:,:date_range]

    current_infected = np.zeros_like(data[1,:])
    dI = data[1,1:]-data[1,:-1]
    for i in range(days_infected):
        current_infected[i] = np.sum(dI[:i])
    for i in range(days_infected,date_range):
        current_infected[i] = np.sum(dI[i-days_infected:i])
    data[1,:] = current_infected[:]
    return data


if __name__ == '__main__':
    data = extract_data('France_Data.csv',14,500)
    plt.plot(data[0,:500],data[1,:500],'s',label = 'Infected')
    #plt.semilogy(data[0,:200],data[2,:200],'s',label = 'Deaths')
    #plt.semilogy(data[0,:200],data[3,:200],'s',label = 'Recovered')
    plt.legend()
    plt.show()

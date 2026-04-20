import numpy as np

import matplotlib.pyplot as plt


class Graphic:
    
    def plot_scatter(x, y, title, xlabel, ylabel, file_name):
        unique, counts = Graphic.__count_points(x, y)
        plt.figure(figsize=(7, 5))
        plt.scatter(unique[:, 0], unique[:, 1], s=30, facecolors='none', edgecolors='blue', linewidths=counts)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        plt.close() 


    def __count_points(x, y):
        array = [[x[i], y[i]] for i in range(len(x))]
        unique, counts = np.unique(array, axis=0, return_counts=True)
        return (unique, counts)
    
    
    def plot_graphic(x, y, title, xlabel, ylabel, file_name):
        plt.figure(figsize=(7, 5))
        plt.plot(x, y,  color='black', lw=0.7)
        #plt.scatter(x, y,  facecolor="none", edgecolor='black', marker="p")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        plt.close() 
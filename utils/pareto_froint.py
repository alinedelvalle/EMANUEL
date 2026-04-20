import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pymoo.indicators.hv import HV


class Point:
    
    def __init__(self, obj1, obj2, norm, meka, weka, k = None, predictions=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.norm = norm
        self.meka = meka
        self.weka = weka
        # predictions
        self.k = k
        self.predictions = predictions
        # algorithm
        self.n = 0
        self.rank = 0
        self.S = []
        
        
    def dominate(self, pto):
        flag = False
        
        if (self.obj1 <= pto.obj1 and self.obj2 <= pto.obj2):
            if (self.obj1 < pto.obj1 or self.obj2 < pto.obj2):
                flag = True
                
        return flag
    
    
    def __str__(self):
        #return self.norm+', '+self.meka+', '+self.weka+', '+str(self.obj1)+', '+str(self.obj2)
        return str(self.obj1)+', '+str(self.obj2)
    
    def __repr__(self):        
        return repr((self.obj2, self.obj1))
   

# fast non dominated sort        
class FNDS:
    
    def __init__(self):
        self.froint = np.array([])
    
    
    def execute(self, list_points):        
        first_froint = []

        # elimina pontos duplicados na lista de pontos
        list_points = self._get_unique_points(list_points)
        
        for i in range(len(list_points)):
            p = list_points[i]
            p.S = []
            p.n = 0
            
            for j in range(len(list_points)):
                
                if i != j:
                    q = list_points[j]

                    if p.dominate(q):
                        p.S.append(q)
                    elif q.dominate(p):
                        p.n += 1

            if p.n == 0:
                p.rank = 1
                first_froint.append(p)
        
        self.froint = first_froint
        
        return self.froint


    def _get_unique_points(self, list_points):
        dic_points = {}
        
        for pto in list_points:
            cmd = str(pto.norm)+' '+pto.meka
            
            if pto.weka != None:
                cmd += ' '+pto.weka
                
            dic_points[cmd] = pto

        list_points_unique = np.array(list(dic_points.values()))
        
        return list_points_unique
    
    
    def get_obj1_from_front(self):
        list_obj1 = np.array([])
        for pto in self.froint:
            list_obj1 = np.append(list_obj1, pto.obj1)
        return list_obj1
        
    
    def get_obj2_from_front(self):
        list_obj2 = np.array([])
        for pto in self.froint:
            list_obj2 = np.append(list_obj2, pto.obj2)
        return list_obj2
    
    
    def get_data_from_front(self):
        list_data = np.array([])
        for pto in self.froint:
            algorithm_data = [pto.norm, pto.meka, pto.weka, -pto.obj1, pto.obj2]
            if len(list_data)==0:
                list_data = np.array(algorithm_data)
            else:
                list_data = np.vstack((list_data, algorithm_data))
        return list_data
    
    
    # select Pareto points 
    def choose_point_by_frugality(self):
        w = 0.5
        
        # penaliza -(-f1) com model size
        all_obj1 = -self.get_obj1_from_front()
        all_obj2 = self.get_obj2_from_front()
        
        min_obj2 = all_obj2.min()
        max_obj2 = all_obj2.max()
        
        all_obj2_norm = (all_obj2 - min_obj2)/(max_obj2 - min_obj2)
        # intervalo de 0.01 a 1 para evitar divisão por 0
        all_obj2_norm = all_obj2_norm * (1 - 0.01) + 0.01 
        
        frugality = np.zeros(len(all_obj2))
        for i in range(len(all_obj2)):     
            frugality[i] = all_obj1[i] - w/(1 + 1/all_obj2_norm[i])
        
        i = np.argmax(frugality)
        
        return self.froint[i]
    

    # min model size e f-score
    def choose_point_by_min_obj2(self): 
        all_model_size = self.get_obj2_from_front()
        i = np.argmin(all_model_size)
        return self.froint[i]
        
    
    # max model size e f-score
    def choose_point_by_max_obj2(self): 
        all_model_size = self.get_obj2_from_front()
        i = np.argmax(all_model_size)
        return self.froint[i]
    
    def get_hypervolume(self, point):            
        metric = HV (ref_point = point)

        F = []
        for pto in self.froint:
            F.append([pto.obj1, pto.obj2])

        hv = metric.do(np.array(F))
            
        return hv
    
    # plots
    def plot_points_and_froint(self, list_points, title, xlabel, ylabel, file_name):
        x_all = []
        y_all = []
        for pto in list_points:
            x_all.append(pto.obj1)
            y_all.append(pto.obj2)
            
        x = []
        y = []
        for pto in self.froint:
            x.append(pto.obj1)
            y.append(pto.obj2)
            
        plt.figure(figsize=(7, 5))
        plt.scatter(x_all, y_all, s=30, facecolors='none', edgecolors='gray')
        plt.scatter(x, y, s=30, facecolors='none', edgecolors='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        plt.close() 
        
        
    def plot_froint(self, title, xlabel, ylabel, file_name):            
        x = []
        y = []
        for pto in self.froint:
            x.append(pto.obj1)
            y.append(pto.obj2)
            
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, s=30, edgecolors='black', color='black')#facecolors='none', edgecolors='blue')
        #plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        plt.close() 
        
        
    # Para artigo
    def plot_teste(self, all_frontiers, xlabel, ylabel, file_name):
        plt.figure(figsize=(7, 5))

        for f in all_frontiers:
            x = []
            y = []
            froint = f[0]
            #my_froint = self.execute(f[0])
            #my_froint = sorted(my_froint, key=lambda pto: (pto.obj2), reverse=True) 

            for pto in froint:
                x.append(pto.obj1)
                y.append(pto.obj2)
                
            plt.scatter(x, y, s=20, edgecolors='gray', color='gray')
            
                
        x = []
        y = []
        my_froint = self.froint.copy()
        my_froint = self.execute(my_froint)
        my_froint = sorted(my_froint, key=lambda pto: (pto.obj2), reverse=True)
        
        for pto in my_froint:
            x.append(pto.obj1)
            y.append(pto.obj2)
            
        plt.plot(x, y, linewidth=2, color='black') 
        plt.scatter(x, y, s=20, edgecolors='black', color='black')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        plt.close() 

        
    def plot_froint_and_choose(self, selected_x, selected_y, title, xlabel, ylabel, file_name):            
        '''
        x = []
        y = []
        for pto in self.froint:
            x.append(pto.obj1)
            y.append(pto.obj2)
            
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, s=30, edgecolors='gray', color='gray') #edgecolors='blue')
        plt.scatter(selected_x, selected_y, s=30, edgecolors='red', color='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        
        '''
        x = []
        y = []
        for pto in self.froint:
            x.append(pto.obj1)
            y.append(pto.obj2)
         
        #plt.rcParams.update({'font.size': 14})
        plt.style.use('default')
        plt.figure(figsize=(4, 3))

        for pto in self.froint:
            if (pto.obj1 != selected_x[0] and pto.obj2 != selected_y[0]) and (pto.obj1 != selected_x[1] and pto.obj2 != selected_y[1]) and (pto.obj1 != selected_x[2] and pto.obj2 != selected_y[2]):
                    plt.scatter(pto.obj1, pto.obj2, s=20, edgecolors='gray', c='gray')

        #plt.scatter(x, y, s=20, edgecolors='gray', c='gray')
        plt.scatter(selected_x[0], selected_y[0], s=20, marker='x', label='Min', edgecolors='black', c='black')
        plt.scatter(selected_x[1], selected_y[1], s=20, marker='D', label='Cent', edgecolors='black', c='black')
        plt.scatter(selected_x[2], selected_y[2], s=20, marker='s', label='Max', edgecolors='black', c='black')
        #plt.title(title)
        plt.xlabel('- Macro F-score')
        plt.ylabel('Model Size')
        plt.legend()
        plt.savefig(file_name+'.pdf', bbox_inches='tight')
        plt.close()
        
        

        
        
# teste dominância
if __name__ == '__main__':
    a = Point(-0.1, 5, True, 'a', 'b')
    f = Point(-0.5, 5, False, 'l', 'm')
    b = Point(-0.2, 3, True, 'c', 'd')
    e = Point(-0.4, 3, False, 'i', 'j')
    c = Point(-0.4, 1, True, 'e', 'f')
    d = Point(-0.3, 4, False, 'g', 'h')
    g = Point(-0.25, 2, False, 'n', 'o')
    
    list_points = [a, b, c, d, e, f, g]

    fnds = FNDS()
    froint = fnds.execute(list_points)
    
    for pto in froint:
        print(pto)
    
    fnds.plot_froint('title', 'xlabel', 'ylabel', '/home/dellvale/Testes/Cluster/Experimento2/size-fscore/results-new/froint.png')
        
    # Para testar, mudar: all_obj1 = -self.get_obj1_from_front()
    # para: all_obj1 = self.get_obj1_from_front()
    fnds.choose_point_by_frugality()
        
    '''F = np.array([[1,2,4,3,4,5], [5,3,1,4,3,5]]).transpose()    
    metric = HV(ref_point=np.array([5.0, 5.0]))
    hv = metric.do(F)
    print(F)
    print(hv)'''

    print(fnds.get_hypervolume([5.0, 5.0]))
    
    # ordenação
    new = sorted(list_points, key=lambda pto: (pto.obj1, -pto.obj2), reverse=False) 
    print(new)
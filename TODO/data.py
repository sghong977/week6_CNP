import numpy as np
import matplotlib.pyplot as plt


#-------------------------- 1D DATA ----------------------------

def generate_data(num_data=20, x_range=(-3,3), std=3.):
    # train data    
    x_train = [[np.random.uniform(*x_range)] for _ in range(num_data)]
    y_train = [[x[0]**3 +np.random.normal(0,std)] for x in x_train]

    # test data
    x_test = np.linspace(-6,6,100).reshape(100,1) # test data for regression
    y_test = x_test**3

    return x_train ,y_train, x_test, y_test


def draw_graph(x,y,x_set,y_set,mean_predict,std): # x-s
    plt.plot(x,y,'b-', label = "Ground Truth")
    plt.plot(x_set, y_set,'ro', label = 'data points')
    plt.plot(x, mean_predict, label='MLPs (MSE)', color='grey')
    plt.fill_between(x.reshape(-1), (mean_predict-3*std).reshape(100,), (mean_predict+3*std).reshape(100,),color='grey',alpha=0.3)

    plt.legend()
    plt.savefig("result.png")
    #plt.show()

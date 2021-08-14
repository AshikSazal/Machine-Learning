import numpy as np
import matplotlib.pyplot as plt
def coefficient(x,y):
    # y=mx+c
    n=len(x)
    sum_xy=(np.multiply(x,y)).sum()*n
    sum_x=x.sum()
    sum_y=y.sum()
    sum_x2=(np.square(x)).sum()*n
    slope=(sum_xy-sum_x*sum_y)/(sum_x2-np.square(sum_x)) # Slope(b) = (NΣXY - (ΣX)(ΣY)) / (NΣX2 - (ΣX)2)
    # slope = ((x-x.mean())*(y-y.mean())).sum()/(np.square(x-x.mean())).sum()
    c=y[0]-slope*x[0]
    return c,slope

def vitualization(x,y,theta_0,theta_1):
    plt.scatter(x,y,color='red',s=50)
    predict=theta_0+theta_1*x
    plt.plot(x,predict,color="blue")
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return np.around(predict,5)

if __name__ == '__main__':

    x=np.array([1,2,2,3,4,4,6])
    y=np.array([2,1,3,3,5,4,4])
    # h(x)=theta_0+theta_1*x
    theta_0,theta_1=coefficient(x,y)
    predict=vitualization(x,y,theta_0,theta_1)
    cost=np.around(((np.square(predict-y)).sum())/(2*len(y)),5)
    print(cost)

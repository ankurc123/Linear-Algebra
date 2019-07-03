import numpy as np 


class sample(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def pearsonR(self):
        sumX = 0
        sumy = 0
        for i, j in zip(self.X, self.y):
            sumX += i
            mean = sumX/len(self.X)

            sumy += j
            mean = sumy/len(self.y)

        X_list = []
        y_list = []
        for i , j in zip(self.X,self.y):
            covar_x = i - mean
            covar_y = j - mean
            X_list.append(covar_x)
            y_list.append(covar_y)

        coVar = np.multiply(X_list, y_list)

        sumCoVar = 0 
        for i in coVar:
            sumCoVar += i
        return sumCoVar


    def slope(self):
        stdX = np.std(self.X)
        stdY = np.std(self.y)

        m = self.pearsonR()*(stdY/stdX)

        return m


    def yIntercept(self):
        y_ = np.mean(self.y) - self.slope() * np.mean(self.X) 
        return y_

    def simpleLinearRegression(self, X):
        return np.dot(self.slope(),X) + self.yIntercept()

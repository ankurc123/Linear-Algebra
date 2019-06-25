import numpy as np

class simpleLinearRegression:
    """the formula of linear regression is 
    y = Wx+b where m and b are the weight and bias respectively and x
    is the input variable. 
    
    Weight or the slope as it is reffered to in the formula of straight line 
    is nothing but the formula W = r (pearson) multiplied by the 
    sum of the (X - Mean of X) multplied by (y - mean of y)
    whole divided by variance of (X - Mean of X) multiplied by  (y - mean of y)

    W =   r * S( (X-mean(X)) * (y-mean(y))) 
        -------------------------------------
        Sqrt of (S(Variance(X) * Variance(y)))
    
    """
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def r(self):
        mean_x = 0
        mean_y = 0
        for i,j in zip(self.x,self.y):
            mean_x += i
            meanOfX = mean_x/len(self.x)
            mean_y += j
            meanOfy = mean_y/len(self.y)

        xx_ = []
        yy_ = []
        for i, j in zip(self.x,self.y):
            x_x = i - meanOfX
            xx_.append(x_x)

            y_y = j - meanOfy
            yy_.append(y_y)

        sum_xx_ = 0
        sum_yy_ = 0
        for i,j in zip(xx_, yy_):
            sum_xx_ += i
            sum_yy_ += j

        var_x = (sum_xx_)**2
        var_y = (sum_yy_)**2

        r = ((sum_xx_*sum_yy_ )/ np.sqrt(var_x*var_y))
        
        return r
    

    def weight(self):
        S_x = np.std(self.x)
        S_y = np.std(self.y)

        w = self.r() * (S_y/S_x)
        return w


    def bias(self):
        y_ = np.mean(self.y)
        b = y_ - np.dot(self.weight(),np.mean(self.x))
        return b

    def simpleLinearRegression(self):
        wx = np.dot(self.weight(),self.x)
        y = wx+self.bias()
        return y 

    def predict(self, X_test):
        wx = np.dot(self.weight(),X_test)
        y = wx+self.bias()
        return y 


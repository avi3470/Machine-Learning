#Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasetset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values 
y = dataset.iloc[:,2].values 
'''x = [0,1,2,3,4,5,6,7,8,9]
x = x.array(x).reshape(len(x),1)
x = scaler.transform(x)'''
"""from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])"""
# no trining and no test set required because here we want to predict actual salary of the new person so we can avoid 
 #fitting linear Regression to the dataset 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
 
 #fitting Poly regression to thr data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x) 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)
  
#visualizing the  Linear Reg
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('truth or bluff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()
#visualizing the  Polynimial Reg
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title('truth or bluff (Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#predicting a new result with Linear reg
lin_reg.predict([[6.5]])


#predicting a new result with Ploynomial reg

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))












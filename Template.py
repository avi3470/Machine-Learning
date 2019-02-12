#regression template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasetset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values 
y = dataset.iloc[:,2].values 


#fitting  regression model to thr data set
#here we will create future non regression model 

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x,y)



#predicting a new result 

y_pred = regressor.predict(([[6.5]]))


#visualizing the   Regression model
plt.scatter(x,y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('truth or bluff ( regression model)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()



#visualizing the   Regression model for heigher resolution and smoother curve
x_grid = np.arange(min(x),max(x),0.1)# this will aslo predict 90  imaginary level 
 #this will gives us vector and we need matrix so so we have to use reahpe function
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('truth or bluff ( regression model)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()








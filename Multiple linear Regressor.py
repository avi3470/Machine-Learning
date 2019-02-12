import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasetset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1] .values 
y = dataset.iloc[:,4].values 

"""from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x =LabelEncoder() #object of labelencoder class is created
x[:,3]=labelencoder_x.fit_transform(x[:,3]) #fitting the coloum to labelcoder obj to encode kaunsa coloum encode karna hai usko bata do
onehotencoder = OneHotEncoder(categorical_features = [3])
x=onehotencoder.fit_transform(x).toarray()
#y variabe need not to be encoded because it is only one variable type
#avoiding dummy variable trap
x=x[:,1:]
from sklearn.model_selection import train_test_split #earlier we use cross_validation lib now model_selection
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state =0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set result
y_pred = regressor.predict(x_test)

#building the optimal model using backward Elimination
import statsmodels.formula.api as sm
#our library does not account of b0 constent term so manually we have to take care of that 
#so we add to add a coloum in matrix of feature x which cantain all 1
x = np.append(arr= np.ones((50,1)).astype(int),values= x,axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog = x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog = x_opt).fit()
regressor_OLS.summary()














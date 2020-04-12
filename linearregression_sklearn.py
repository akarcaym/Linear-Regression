import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#reading the data
data=pd.read_csv('headbrain.csv')
data.head()
x=data.iloc[:,2:3].values
y=data.iloc[:,3:4].values

#splitting the data into training and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

#fitting simple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predict the test result
y_pred = regressor.predict(x_test)

#to see the relationship between the training data values
plt.scatter(x_train,y_train,c='red')
plt.show()

#to see the relationship between the predicted
#brain weight values using scattered graph
plt.plot(x_test,y_pred)
plt.scatter(x_test,y_test,c='red')
plt.xlabel('headsize')
plt.ylabel('brain weight')
plt.show()

#error in each value
for i in range(0,60):
    print("Error in value number",i,(y_test[i]-y_pred[i]))
time.sleep(1)

#combined rmse value
rss=((y_test-y_pred)**2).sum()
mse=np.mean((y_test-y_pred)**2)
print("RMSE=",np.sqrt(np.mean((y_test-y_pred)**2)))
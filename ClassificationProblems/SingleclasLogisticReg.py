'''Implement the machine learning to predict whether the people are interested to take insurance
not '''
'''a) With data splitting '''
import pandas as pd
df = pd.read_csv("insurance_data.csv")
print(df)
print("checking the null if any column have")
print(df.isnull())
print("listing the is null values into the groups")
print(df.isnull().sum())
'''Now Ploting the data using matplotlib '''
import matplotlib.pyplot as plt
plt.scatter(df.age,df.bought_insurance,color="red")
plt.show()
'''Now i am going to the use Logistics Regression instead of the Linear because
I have seen the graph which shows that its values of the data is not
going the linear for seprations and so that instead of using the Linear Regressions for 
the bettter i have to use the Logistics Regressions Okay???'''
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(df[["age"]],df.bought_insurance,test_size=0.1,random_state=0)
# print("the values of the train_x is "+ "  "+ train_x)
# print("the value of the test_x is "+ " " + test_x)
# print("the value of the train_y is" + " " + train_y)
# print("the value of the test_y is " + " " + test_y)
print("value of the train_x is ")
print(train_x)
print("the value of the test_x is ")
print(test_x)
print("the value of the train_y is ")
print(train_y)
print("the value of the test_y is ")
print(test_y)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_x,train_y)
print(model.predict(test_x))
print("the new data is (78 year for the insurance either the people are interested take insurance)")
print(model.predict([[78]]))
print(model.score(test_x,test_y))



'''Implementations of machine learning models to check whether the peoples are interested in 
to bought insurance or not by using the age
'''
'''Without Splitting the Datasets '''
print("<--------------printing the datasets  Without Splitting Datasets----------------->")
import pandas as pd
df = pd.read_csv("insurance_data.csv")
print(df)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(df[["age"]],df.bought_insurance)
print("printing the new pridiction of the data by using the value 56")
print(model.predict([[56]]))
print("again predicting for the people whose age is 90")
print(model.predict([[90]]))
print("again predicting for the people whose age is 3")
print(model.predict([[3]]))


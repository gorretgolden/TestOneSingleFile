#!/usr/bin/env python
# coding: utf-8

# # Test One

# <h4>Importing Libraries</h4>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# <h4>Question0</h4>

# In[3]:


#one dimentional array
a = np.array([1,1,1])
a


# <h4>Question1</h4>
# 

# In[2]:


#let the matrix be m
m = np.array([[1,0.2,0.5],[0.2,1,0.8],[0.5,0.8,1]])
m


# In[4]:


#transpose of m
print('Transpose of m is:',m.transpose())


# In[18]:


#let determinant of m be det
det = np.linalg.det(m)
print('Determinant of m is:',det)


# <h4>Question2</h4>

# In[5]:


#dataframe from the matrix, let the dataframe be df
df =pd.DataFrame(m,columns=['A', 'B', 'C'],index=['A', 'B', 'C'])
df


# <h4>Question3</h4>

# In[6]:


#Standard Deviation of the data, assuming the data is an array
data = np.array([1,3,1,2,9,4,5,6,10,4])
 
standard_deviation = np.std(data)
standard_deviation


# <h4>Question4</h4>

# In[8]:


#Function to return a user value

def  QUESTION_4():
    x = input('Enter a number')
    return x

print( " The value of x is:",QUESTION_4())
   
  


# <h4>Question5</h4>

# In[9]:


#Covid 19 Results
covidData = pd.read_csv(r'C:\Users\golden\Desktop\witiAcademy\Data Science\Test1\COVID-19 Cases.csv')
covidData


# In[10]:


#convert the column (it's a string) to datetime type
datetime_series = pd.to_datetime(covidData['Date'])

# create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values)

# Setting the index, df is the new dataFrame
df = covidData.set_index(datetime_index)


# In[11]:


#Printing the first 10 
df.head(10)


# In[ ]:





# <h4>Question6</h4>

# In[12]:


#Data filtering using boolean indexing
df_results = df[(df.Difference >0) & (df.Case_Type == 'Confirmed') & (df.Country_Region == 'Italy')]
df_results


# In[82]:


#Sorting the values
df_results.sort_values(by=['Cases'],ascending=True)


# <h4>Question7</h4>

# In[19]:


#plotting the histogram using the Difference Column

histogram = df.Difference
histogram.hist()
# plt.show() 


# <h4>Question8</h4>

# In[20]:


#Data Description
df.describe()


# In[21]:


#New dataFrame, let it be new_df
new_df = df[['Difference','Country_Region','Cases']]
new_df


# In[86]:


#Looking at the Country_Region column
new_df['Country_Region']


# <h4>Question9</h4>

# In[22]:


new_df.boxplot(by='Country_Region', column=['Difference'], grid=False);


# <h4>Question10</h4>

# In[23]:


# new dataframe
covidData2= pd.read_csv(r'C:\Users\golden\Desktop\witiAcademy\Data Science\Test1\COVID-19 Cases.csv')
covidData2


# <h4>Question11</h4>

# In[96]:


#Country region, germany
df_results_1 = covidData2[(covidData2.Country_Region == 'Germany' ) & (covidData2.Case_Type == 'Confirmed' ) ]
df_results_1


# In[98]:


#Country region, Italy
df_results_2 = covidData2[(covidData2.Country_Region == 'Italy' ) & (covidData2.Case_Type == 'Confirmed' ) ]
df_results_2


# <h4>Question12</h4>

# In[111]:


#Scatter plot
covidData2.plot.scatter(x = 'Country_Region', y = 'Cases')
plt.xlabel('Country')
plt.ylabel('Cases')
plt.title('Scatter Plot for Countries and the Covid-19 Cases')
plt.show()


# In[ ]:





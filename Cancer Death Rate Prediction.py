#!/usr/bin/env python
# coding: utf-8

# # ***`CANCER DEATH RATE PREDICTION`***

# #### `IMPORT REQUIRED PACKAGES(LIBRARIES)`

# In[ ]:


import pyforest
import warnings
warnings.filterwarnings("ignore")
from matplotlib.cm import get_cmap


# ***pyforest*** consists of all popular Python Data Science libraries which should account for >99% of your daily imports. For example, **pandas as pd**, **numpy as np**, **seaborn as sns**, **matplotlib.pyplot as plt**, or **OneHotEncoder from sklearn** and many more. In addition, there are also helper modules like **os, re, tqdm, or Path from pathlib**.

# `Set up display area to show dataframe in jupyter console`

# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# #### `READ THE DATASET as df`

# In[ ]:


df = pd.read_csv('cancer_mortality.csv')


# #### `CONFIRM THE DATA TRANSFER FROM LOCAL DRIVE TO NOTEBOOK`

# In[ ]:


print('First Five Records of the Dataset')
df.head()  # As default shows the top 5 Rows.


# In[ ]:


print('Last Five Records of the Dataset')
df.tail()  # As default shows the Bottom 5 Rows.


# #### `DATA UNDERSTANDING`

# `Check for the Shape`

# In[ ]:


print("The Data Frame having the Rows of '{}' and Columns of '{}'".format (df.shape[0],df.shape[1]))


# `Check for the Detailed Information of the Dataset`

# In[ ]:


print('Total_Columns: ', len(df.columns),'\n')
print(df.columns,'\n')
print('Shape :',df.shape)


# In[ ]:


df.info()


# `Check for the Null values in the Dataset`

# In[ ]:


df.isnull().sum()


# `Statistical Information :`

# In[ ]:


df.describe()


# `Displays memory consumed by each column`

# In[ ]:


print(df.memory_usage(),'\n')
print('Dataset uses {0} MB'.format(df.memory_usage().sum()/1024**2))


# `Number of unique values in each column`

# In[ ]:


def unique(x):
    return len(df[x].unique())

number_unique_vals = {x: unique(x) for x in df.columns}
number_unique_vals


# #### `UNIVARIATE ANALYSIS & VISUALIZATION`

# ***`avgAnnCount`*** :  Mean number of reported cases of cancer diagnosed annually

# In[ ]:


print('Variable_name : ' ,df.iloc[:,0].name)
print('Type : ',df.iloc[:,0].dtype)

print('Null_value_count: ',df.iloc[:,0].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,0].skew())
df.iloc[:,0].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,0], color = 'green')
plt.xlabel(df.iloc[:,0].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,0].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,0], color = 'orange')
plt.xlabel(df.iloc[:,0].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,0].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower limit - 5% :', df.iloc[:,0].quantile(0.05),'\n Upper limit - 95% :', df.iloc[:,0].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,0] = np.where(df.iloc[:,0] > df.iloc[:,0].quantile(0.95), df.iloc[:,0].quantile(0.95), df.iloc[:,0])
df.iloc[:,0].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,0], color = 'green')
plt.xlabel(df.iloc[:,0].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,0].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,0], color = 'orange')
plt.xlabel(df.iloc[:,0].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,0].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`avgDeathsPerYear`*** :  Mean number of reported mortalities due to cancer

# In[ ]:


print('Variable_name : ' ,df.iloc[:,1].name)
print('Type : ',df.iloc[:,1].dtype)

print('Null_value_count: ',df.iloc[:,1].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,1].skew())
df.iloc[:,1].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,1], color = 'green')
plt.xlabel(df.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,1], color = 'orange')
plt.xlabel(df.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,1].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,1].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,1] = np.where(df.iloc[:,1] > df.iloc[:,1].quantile(0.95), df.iloc[:,1].quantile(0.95), df.iloc[:,1])
df.iloc[:,1].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,1], color = 'green')
plt.xlabel(df.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,1], color = 'orange')
plt.xlabel(df.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`TARGET_deathRate`*** :  Dependent variable. Mean per capita (100,000) cancer mortalities

# In[ ]:


print('Variable_name : ' ,df.iloc[:,2].name)
print('Type : ',df.iloc[:,2].dtype)

print('Null_value_count: ',df.iloc[:,2].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,2].skew())
df.iloc[:,2].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,2], color = 'green')
plt.xlabel(df.iloc[:,2].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,2].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,2], color = 'orange')
plt.xlabel(df.iloc[:,2].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,2].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`incidenceRate`*** :  Mean per capita (100,000) cancer diagnoses

# In[ ]:


print('Variable_name : ' ,df.iloc[:,3].name)
print('Type : ',df.iloc[:,3].dtype)

print('Null_value_count: ',df.iloc[:,3].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,3].skew())
df.iloc[:,3].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,3], color = 'green')
plt.xlabel(df.iloc[:,3].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,3].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,3], color = 'orange')
plt.xlabel(df.iloc[:,3].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,3].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,3].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,3].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,3] = np.where(df.iloc[:,3] < df.iloc[:,3].quantile(0.05), df.iloc[:,3].quantile(0.05), df.iloc[:,3])
df.iloc[:,3].describe()


# In[ ]:


df.iloc[:,3] = np.where(df.iloc[:,3] > df.iloc[:,3].quantile(0.95), df.iloc[:,3].quantile(0.95), df.iloc[:,3])
df.iloc[:,3].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,3], color = 'green')
plt.xlabel(df.iloc[:,3].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,3].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,3], color = 'orange')
plt.xlabel(df.iloc[:,3].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,3].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`medIncome`*** :  Median income per county

# In[ ]:


print('Variable_name : ' ,df.iloc[:,4].name)
print('Type : ',df.iloc[:,4].dtype)

print('Null_value_count: ',df.iloc[:,4].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,4].skew())
df.iloc[:,4].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,4], color = 'green')
plt.xlabel(df.iloc[:,4].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,4].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,4], color = 'orange')
plt.xlabel(df.iloc[:,4].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,4].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,4].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,4].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,4] = np.where(df.iloc[:,4] > df.iloc[:,4].quantile(0.95), df.iloc[:,4].quantile(0.95), df.iloc[:,4])
df.iloc[:,4].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,4], color = 'green')
plt.xlabel(df.iloc[:,4].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,4].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,4], color = 'orange')
plt.xlabel(df.iloc[:,4].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,4].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`popEst2015`*** :  Population of county

# In[ ]:


print('Variable_name : ' ,df.iloc[:,5].name)
print('Type : ',df.iloc[:,5].dtype)

print('Null_value_count: ',df.iloc[:,5].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,5].skew())
df.iloc[:,5].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,5], color = 'green')
plt.xlabel(df.iloc[:,5].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,5].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,5], color = 'orange')
plt.xlabel(df.iloc[:,5].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,5].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,5].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,5].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,5] = np.where(df.iloc[:,5] > df.iloc[:,5].quantile(0.95), df.iloc[:,5].quantile(0.95), df.iloc[:,5])
df.iloc[:,5].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,5], color = 'green')
plt.xlabel(df.iloc[:,5].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,5].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,5], color = 'orange')
plt.xlabel(df.iloc[:,5].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,5].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`povertyPercent`*** :  Percent of population in poverty

# In[ ]:


print('Variable_name : ' ,df.iloc[:,6].name)
print('Type : ',df.iloc[:,6].dtype)

print('Null_value_count: ',df.iloc[:,6].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,6].skew())
df.iloc[:,6].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,6], color = 'green')
plt.xlabel(df.iloc[:,6].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,6].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,6], color = 'orange')
plt.xlabel(df.iloc[:,6].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,6].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`studyPerCap`*** :  Per capita number of cancer-related clinical trials per county

# In[ ]:


print('Variable_name : ' ,df.iloc[:,7].name)
print('Type : ',df.iloc[:,7].dtype)

print('Null_value_count: ',df.iloc[:,7].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,7].skew())
df.iloc[:,7].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,7], color = 'green')
plt.xlabel(df.iloc[:,7].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,7].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,7], color = 'orange')
plt.xlabel(df.iloc[:,7].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,7].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,7].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,7].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,7] = np.where(df.iloc[:,7] > df.iloc[:,7].quantile(0.95), df.iloc[:,7].quantile(0.95), df.iloc[:,7])
df.iloc[:,7].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,7], color = 'green')
plt.xlabel(df.iloc[:,7].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,7].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,7], color = 'orange')
plt.xlabel(df.iloc[:,7].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,7].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`binnedInc`*** :  Median income per capita binned by decile

# In[ ]:


print('Variable_name : ' ,df.iloc[:,8].name)
print('Type : ',df.iloc[:,8].dtype)

print('Null_value_count: ',df.iloc[:,8].isna().sum())


# In[ ]:


df.iloc[:,8].head()


# `Removed the special characters`

# In[ ]:


df = df.drop(df[['binnedInc']],axis = 1)


# ***`MedianAge`*** :  Median age of county residents

# In[ ]:


print('Variable_name : ' ,df.iloc[:,8].name)
print('Type : ',df.iloc[:,8].dtype)

print('Null_value_count: ',df.iloc[:,8].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,8].skew())
df.iloc[:,8].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,8], color = 'green')
plt.xlabel(df.iloc[:,8].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,8].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,8], color = 'orange')
plt.xlabel(df.iloc[:,8].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,8].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,8].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,8].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,8] = np.where(df.iloc[:,8] > df.iloc[:,8].quantile(0.95), df.iloc[:,8].quantile(0.95), df.iloc[:,8])
df.iloc[:,8].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,8], color = 'green')
plt.xlabel(df.iloc[:,8].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,8].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,8], color = 'orange')
plt.xlabel(df.iloc[:,8].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,8].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`MedianAgeMale`*** :  Median age of male county residents

# In[ ]:


print('Variable_name : ' ,df.iloc[:,9].name)
print('Type : ',df.iloc[:,9].dtype)

print('Null_value_count: ',df.iloc[:,9].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,9].skew())
df.iloc[:,9].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,9], color = 'green')
plt.xlabel(df.iloc[:,9].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,9].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,9], color = 'orange')
plt.xlabel(df.iloc[:,9].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,9].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`MedianAgeFemale`*** :  Median age of female county residents

# In[ ]:


print('Variable_name : ' ,df.iloc[:,10].name)
print('Type : ',df.iloc[:,10].dtype)

print('Null_value_count: ',df.iloc[:,10].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,10].skew())
df.iloc[:,10].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,10], color = 'green')
plt.xlabel(df.iloc[:,10].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,10].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,10], color = 'orange')
plt.xlabel(df.iloc[:,10].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,10].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`Geography`*** :  County name

# In[ ]:


print('Variable_name : ' ,df.iloc[:,11].name)
print('Type : ',df.iloc[:,11].dtype)

print('Null_value_count: ',df.iloc[:,11].isna().sum())


# In[ ]:


df.iloc[:,11] = df.iloc[:,11].str.lower()


# In[ ]:


df.iloc[:,11].head()


# In[ ]:


df.iloc[:,11].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


df = df.drop(df[['Geography']],axis = 1)


# ***` AvgHouseholdSize`*** :  Average rate of house hold members

# In[ ]:


print('Variable_name : ' ,df.iloc[:,11].name)
print('Type : ',df.iloc[:,11].dtype)

print('Null_value_count: ',df.iloc[:,11].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,11].skew())
df.iloc[:,11].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,11], color = 'green')
plt.xlabel(df.iloc[:,11].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,11].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,11], color = 'orange')
plt.xlabel(df.iloc[:,11].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,11].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PercentMarried`*** :  Percent of county residents who are married

# In[ ]:


print('Variable_name : ' ,df.iloc[:,12].name)
print('Type : ',df.iloc[:,12].dtype)

print('Null_value_count: ',df.iloc[:,12].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,12].skew())
df.iloc[:,12].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,12], color = 'green')
plt.xlabel(df.iloc[:,12].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,12].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,12], color = 'orange')
plt.xlabel(df.iloc[:,12].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,12].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctNoHS18_24`*** :  Percent of county residents ages 18-24 highest education attained: less than high school

# In[ ]:


print('Variable_name : ' ,df.iloc[:,13].name)
print('Type : ',df.iloc[:,13].dtype)

print('Null_value_count: ',df.iloc[:,13].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,13].skew())
df.iloc[:,13].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,13], color = 'green')
plt.xlabel(df.iloc[:,13].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,13].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,13], color = 'orange')
plt.xlabel(df.iloc[:,13].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,13].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctHS18_24`*** :  Percent of county residents ages 18-24 highest education attained: high school diploma

# In[ ]:


print('Variable_name : ' ,df.iloc[:,14].name)
print('Type : ',df.iloc[:,14].dtype)

print('Null_value_count: ',df.iloc[:,14].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,14].skew())
df.iloc[:,14].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,14], color = 'green')
plt.xlabel(df.iloc[:,14].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,14].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,14], color = 'orange')
plt.xlabel(df.iloc[:,14].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,14].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctSomeCol18_24`*** :  Percent of county residents ages 18-24 highest education attained: some college

# In[ ]:


print('Variable_name : ' ,df.iloc[:,15].name)
print('Type : ',df.iloc[:,15].dtype)

print('Null_value_count: ',df.iloc[:,15].isna().sum())


# In[ ]:


df = df.drop(df[['PctSomeCol18_24']], axis = 1)


# ***`PctBachDeg18_24`*** :  Percent of county residents ages 18-24 highest education attained: bachelor's degree

# In[ ]:


print('Variable_name : ' ,df.iloc[:,15].name)
print('Type : ',df.iloc[:,15].dtype)

print('Null_value_count: ',df.iloc[:,15].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,15].skew())
df.iloc[:,15].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,15], color = 'green')
plt.xlabel(df.iloc[:,15].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,15].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,15], color = 'orange')
plt.xlabel(df.iloc[:,15].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,15].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctHS25_Over`*** :  Percent of county residents ages 25 and over highest education attained: high school diploma

# In[ ]:


print('Variable_name : ' ,df.iloc[:,16].name)
print('Type : ',df.iloc[:,16].dtype)

print('Null_value_count: ',df.iloc[:,16].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,16].skew())
df.iloc[:,16].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,16], color = 'green')
plt.xlabel(df.iloc[:,16].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,16].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,16], color = 'orange')
plt.xlabel(df.iloc[:,16].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,16].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctBachDeg25_Over`*** :  Percent of county residents ages 25 and over highest education attained: bachelor's degree

# In[ ]:


print('Variable_name : ' ,df.iloc[:,17].name)
print('Type : ',df.iloc[:,17].dtype)

print('Null_value_count: ',df.iloc[:,17].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,17].skew())
df.iloc[:,17].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,17], color = 'green')
plt.xlabel(df.iloc[:,17].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,17].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,17], color = 'orange')
plt.xlabel(df.iloc[:,17].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,17].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctEmployed16_Over`*** :  Percent of county residents ages 16 and over employed

# In[ ]:


print('Variable_name : ' ,df.iloc[:,18].name)
print('Type : ',df.iloc[:,18].dtype)

print('Null_value_count: ',df.iloc[:,18].isna().sum())


# In[ ]:


df['PctEmployed16_Over'] = df['PctEmployed16_Over'].fillna(df['PctEmployed16_Over'].mode()[0])


# In[ ]:


df.iloc[:,18].isna().sum()


# In[ ]:


df.iloc[:,18].head()


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,18].skew())
df.iloc[:,18].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,18], color = 'green')
plt.xlabel(df.iloc[:,18].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,18].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,18], color = 'orange')
plt.xlabel(df.iloc[:,18].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,18].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctUnemployed16_Over`*** :  Percent of county residents ages 16 and over unemployed 

# In[ ]:


print('Variable_name : ' ,df.iloc[:,19].name)
print('Type : ',df.iloc[:,19].dtype)

print('Null_value_count: ',df.iloc[:,19].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,19].skew())
df.iloc[:,19].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,19], color = 'green')
plt.xlabel(df.iloc[:,19].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,19].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,19], color = 'orange')
plt.xlabel(df.iloc[:,19].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,19].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctPrivateCoverage`*** :  Percent of county residents with private health coverage

# In[ ]:


print('Variable_name : ' ,df.iloc[:,20].name)
print('Type : ',df.iloc[:,20].dtype)

print('Null_value_count: ',df.iloc[:,20].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,20].skew())
df.iloc[:,20].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,20], color = 'green')
plt.xlabel(df.iloc[:,20].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,20].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,20], color = 'orange')
plt.xlabel(df.iloc[:,20].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,20].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***` PctPrivateCoverageAlone`*** :  Percent of county residents with private health coverage alone (no public assistance)

# In[ ]:


print('Variable_name : ' ,df.iloc[:,21].name)
print('Type : ',df.iloc[:,21].dtype)

print('Null_value_count: ',df.iloc[:,21].isna().sum())


# In[ ]:


df['PctPrivateCoverageAlone'] = df['PctPrivateCoverageAlone'].fillna(df['PctPrivateCoverageAlone'].mode()[0])


# In[ ]:


df.iloc[:,21].isna().sum()


# In[ ]:


df.iloc[:,21].head()


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,21].skew())
df.iloc[:,21].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,21], color = 'green')
plt.xlabel(df.iloc[:,21].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,21].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,21], color = 'orange')
plt.xlabel(df.iloc[:,21].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,21].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctEmpPrivCoverage`*** :  Percent of county residents with employee-provided private health coverage

# In[ ]:


print('Variable_name : ' ,df.iloc[:,22].name)
print('Type : ',df.iloc[:,22].dtype)

print('Null_value_count: ',df.iloc[:,22].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,22].skew())
df.iloc[:,22].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,22], color = 'green')
plt.xlabel(df.iloc[:,22].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,22].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,22], color = 'orange')
plt.xlabel(df.iloc[:,22].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,21].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctPublicCoverage`*** :  Percent of county residents with government-provided health coverage

# In[ ]:


print('Variable_name : ' ,df.iloc[:,23].name)
print('Type : ',df.iloc[:,23].dtype)

print('Null_value_count: ',df.iloc[:,23].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,23].skew())
df.iloc[:,23].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,23], color = 'green')
plt.xlabel(df.iloc[:,23].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,23].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,23], color = 'orange')
plt.xlabel(df.iloc[:,23].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,23].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctPublicCoverageAlone`*** :  Percent of county residents with government-provided health coverage alone

# In[ ]:


print('Variable_name : ' ,df.iloc[:,24].name)
print('Type : ',df.iloc[:,24].dtype)

print('Null_value_count: ',df.iloc[:,24].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,24].skew())
df.iloc[:,24].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,24], color = 'green')
plt.xlabel(df.iloc[:,24].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,24].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,24], color = 'orange')
plt.xlabel(df.iloc[:,24].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,24].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctWhite`*** :  Percent of county residents who identify as White

# In[ ]:


print('Variable_name : ' ,df.iloc[:,25].name)
print('Type : ',df.iloc[:,25].dtype)

print('Null_value_count: ',df.iloc[:,25].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,25].skew())
df.iloc[:,25].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,25], color = 'green')
plt.xlabel(df.iloc[:,25].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,25].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,25], color = 'orange')
plt.xlabel(df.iloc[:,25].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,25].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctBlack`*** :  Percent of county residents who identify as Black

# In[ ]:


print('Variable_name : ' ,df.iloc[:,26].name)
print('Type : ',df.iloc[:,26].dtype)

print('Null_value_count: ',df.iloc[:,26].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,26].skew())
df.iloc[:,26].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,26], color = 'green')
plt.xlabel(df.iloc[:,26].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,26].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,26], color = 'orange')
plt.xlabel(df.iloc[:,26].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,26].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctAsian`*** :  Percent of county residents who identify as Asian

# In[ ]:


print('Variable_name : ' ,df.iloc[:,27].name)
print('Type : ',df.iloc[:,27].dtype)

print('Null_value_count: ',df.iloc[:,27].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,27].skew())
df.iloc[:,27].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,27], color = 'green')
plt.xlabel(df.iloc[:,27].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,27].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,27], color = 'orange')
plt.xlabel(df.iloc[:,27].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,27].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,27].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,27].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,27] = np.where(df.iloc[:,27] > df.iloc[:,27].quantile(0.95), df.iloc[:,27].quantile(0.95), df.iloc[:,27])
df.iloc[:,27].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,27], color = 'green')
plt.xlabel(df.iloc[:,27].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,27].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,27], color = 'orange')
plt.xlabel(df.iloc[:,27].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,27].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctOtherRace`*** :  Percent of county residents who identify in a category which is not White, Black, or Asian

# In[ ]:


print('Variable_name : ' ,df.iloc[:,28].name)
print('Type : ',df.iloc[:,28].dtype)

print('Null_value_count: ',df.iloc[:,28].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,28].skew())
df.iloc[:,28].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,28], color = 'green')
plt.xlabel(df.iloc[:,28].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,28].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,28], color = 'orange')
plt.xlabel(df.iloc[:,28].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,28].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# `Check for the Quartile Ranges`

# In[ ]:


print('Lower Limit - 5% :', df.iloc[:,28].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,28].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[ ]:


df.iloc[:,28] = np.where(df.iloc[:,28] > df.iloc[:,28].quantile(0.95), df.iloc[:,28].quantile(0.95), df.iloc[:,28])
df.iloc[:,28].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,28], color = 'green')
plt.xlabel(df.iloc[:,28].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,28].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,28], color = 'orange')
plt.xlabel(df.iloc[:,28].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,28].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`PctMarriedHouseholds`*** :  Percent of married households

# In[ ]:


print('Variable_name : ' ,df.iloc[:,29].name)
print('Type : ',df.iloc[:,29].dtype)

print('Null_value_count: ',df.iloc[:,29].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,29].skew())
df.iloc[:,29].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,29], color = 'green')
plt.xlabel(df.iloc[:,29].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,29].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,29], color = 'orange')
plt.xlabel(df.iloc[:,29].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,29].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ***`BirthRate`*** :  Number of live births relative to number of women in county

# In[ ]:


print('Variable_name : ' ,df.iloc[:,30].name)
print('Type : ',df.iloc[:,30].dtype)

print('Null_value_count: ',df.iloc[:,30].isna().sum())


# `Check for Outliers using Skewness`

# In[ ]:


print('Skewness: ', df.iloc[:,30].skew())
df.iloc[:,30].describe()


# `Box Plot and Histogram plot for checking the Outliers`

# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,30], color = 'green')
plt.xlabel(df.iloc[:,30].name, fontsize = 20)
plt.title('Boxplot_ '+ df.iloc[:,30].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,30], color = 'orange')
plt.xlabel(df.iloc[:,30].name, fontsize = 20)
plt.title('Histogram_ '+ df.iloc[:,30].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `BIVARIATE ANALYSIS & VISUALIZATION`

# `CORRELATION PLOT`

# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[ ]:


# Scatter plot
df.plot(x='TARGET_deathRate', y='BirthRate', kind='scatter')
plt.show()


# In[ ]:


# Scatter plot
df.plot(x='TARGET_deathRate', y='medIncome', kind='scatter')
plt.show()


# In[ ]:


# Scatter plot
df.plot(x='TARGET_deathRate', y='PctBachDeg25_Over', kind='scatter')
plt.show()


# #### `SPLIT THE DATASET INTO TRAIN AND TEST`

# In[ ]:


x = df.loc[:,df.columns != 'TARGET_deathRate']


# In[ ]:


x.head()


# In[ ]:


y = df.loc[:,df.columns == 'TARGET_deathRate']


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### `FIT THE MODELS`

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(x_train,y_train)


# In[ ]:


y_pred = lr.predict(x_test)


# #### `MODEL-1`

# In[ ]:


import statsmodels.api as sm
x_train_sm = x_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x_train_sm = sm.add_constant(x_train_sm)
# create a fitted model in one line
mlm = sm.OLS(y_train,x_train_sm).fit()

# print the coefficients
mlm.params
print(mlm.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x_train.values, i) 
                          for i in range(len(x_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# #### `MODEL-2`

# In[ ]:


x1_train = x_train[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 'povertyPercent', 
                    'studyPerCap', 'MedianAge', 'MedianAgeMale', 'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 
                    'PctHS18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 
                    'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage', 
                    'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctAsian', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]
x1_test = x_test[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 'povertyPercent', 
                    'studyPerCap', 'MedianAge', 'MedianAgeMale', 'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 
                    'PctHS18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 
                    'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage', 
                    'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctAsian', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]


# 'MedianAgeFemale', 'PctBlack' These two varibles are removed because of significant value is more than 0.05.

# In[ ]:


lr1 = lr.fit(x1_train,y_train)


# In[ ]:


y_pred1 = lr1.predict(x1_test)


# In[ ]:


import statsmodels.api as sm
x1_train_sm = x1_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x1_train_sm = sm.add_constant(x1_train_sm)
# create a fitted model in one line
mlm1 = sm.OLS(y_train,x1_train_sm).fit()

# print the coefficients
mlm1.params
print(mlm1.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred1, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x1_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x1_train.values, i) 
                          for i in range(len(x1_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred1, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred1)
r_squared = r2_score(y_test, y_pred1)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# #### `MODEL-3`

# In[ ]:


x2_train = x_train[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 'povertyPercent', 
                    'studyPerCap', 'MedianAge', 'MedianAgeMale', 'PercentMarried', 'PctNoHS18_24', 
                    'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 
                    'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 
                    'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctAsian', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]
x2_test = x_test[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 'povertyPercent', 
                    'studyPerCap', 'MedianAge', 'MedianAgeMale', 'PercentMarried', 'PctNoHS18_24', 
                    'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 
                    'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 
                    'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctAsian', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]


# 'AvgHouseholdSize', 'PctPrivateCoverageAlone', 'PctBachDeg18_24' These varibles are removed because of significant value is more than 0.05.

# In[ ]:


lr2 = lr.fit(x2_train,y_train)


# In[ ]:


y_pred2 = lr2.predict(x2_test)


# In[ ]:


import statsmodels.api as sm
x2_train_sm = x2_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x2_train_sm = sm.add_constant(x2_train_sm)
# create a fitted model in one line
mlm2 = sm.OLS(y_train,x2_train_sm).fit()

# print the coefficients
mlm2.params
print(mlm2.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred2, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x2_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x2_train.values, i) 
                          for i in range(len(x2_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred2, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred2)
r_squared = r2_score(y_test, y_pred2)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# #### `MODEL-4`

# In[ ]:


x3_train = x_train[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctNoHS18_24', 
                    'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 
                    'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 
                    'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]
x3_test = x_test[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctNoHS18_24', 
                    'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 
                    'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 
                    'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]


# 'studyPerCap', 'MedianAge', 'povertyPercent', 'PctAsian' These varibles are removed because of significant value is more than 0.05.

# In[ ]:


lr3 = lr.fit(x3_train,y_train)


# In[ ]:


y_pred3 = lr3.predict(x3_test)


# In[ ]:


import statsmodels.api as sm
x3_train_sm = x3_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x3_train_sm = sm.add_constant(x3_train_sm)
# create a fitted model in one line
mlm3 = sm.OLS(y_train,x3_train_sm).fit()

# print the coefficients
mlm3.params
print(mlm3.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred3, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x3_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x3_train.values, i) 
                          for i in range(len(x3_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred3, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred3)
r_squared = r2_score(y_test, y_pred3)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# #### `MODEL-5`

# In[ ]:


x4_train = x_train[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]
x4_test = x_test[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]


# 'PctPublicCoverageAlone', 'PctWhite', 'PctNoHS18_24', 'PctUnemployed16_Over' These varibles are removed because of significant value is more than 0.05.

# In[ ]:


lr4 = lr.fit(x4_train,y_train)


# In[ ]:


y_pred4 = lr4.predict(x4_test)


# In[ ]:


import statsmodels.api as sm
x4_train_sm = x4_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x4_train_sm = sm.add_constant(x4_train_sm)
# create a fitted model in one line
mlm4 = sm.OLS(y_train,x4_train_sm).fit()

# print the coefficients
mlm4.params
print(mlm4.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred4, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x4_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x4_train.values, i) 
                          for i in range(len(x4_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred4, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred4)
r_squared = r2_score(y_test, y_pred4)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# #### `MODEL-6`

# In[ ]:


x5_train = x_train[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]
x5_test = x_test[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]


# 'medIncome' This varibles is removed because of significant value is more than 0.05.

# In[ ]:


lr5 = lr.fit(x5_train,y_train)


# In[ ]:


y_pred5 = lr5.predict(x5_test)


# In[ ]:


import statsmodels.api as sm
x5_train_sm = x5_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x5_train_sm = sm.add_constant(x5_train_sm)
# create a fitted model in one line
mlm5 = sm.OLS(y_train,x5_train_sm).fit()

# print the coefficients
mlm5.params
print(mlm5.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred5, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x5_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x5_train.values, i) 
                          for i in range(len(x5_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred5, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred5)
r_squared = r2_score(y_test, y_pred5)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# #### `MODEL-6`

# In[ ]:


x5_train = x_train[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]
x5_test = x_test[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds', 'BirthRate']]


# 'medIncome' This varibles is removed because of significant value is more than 0.05.

# In[ ]:


lr5 = lr.fit(x5_train,y_train)


# In[ ]:


y_pred5 = lr5.predict(x5_test)


# In[ ]:


import statsmodels.api as sm
x5_train_sm = x5_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x5_train_sm = sm.add_constant(x5_train_sm)
# create a fitted model in one line
mlm5 = sm.OLS(y_train,x5_train_sm).fit()

# print the coefficients
mlm5.params
print(mlm5.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred5, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x5_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x5_train.values, i) 
                          for i in range(len(x5_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred5, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred5)
r_squared = r2_score(y_test, y_pred5)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


x5_train = pd.DataFrame(x5_train)
y_train = pd.DataFrame(y_train)


# In[ ]:


df1 = pd.concat([x5_train,y_train],axis = 1)


# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(df1.corr(),annot = True)
plt.show()


# #### `MODEL-7`

# In[ ]:


x6_train = x_train[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]
x6_test = x_test[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 
                    'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]


# 'BirthRate' This varibles is remoed because of less correlation with Target_death rate

# In[ ]:


lr6 = lr.fit(x6_train,y_train)


# In[ ]:


y_pred6 = lr6.predict(x6_test)


# In[ ]:


import statsmodels.api as sm
x6_train_sm = x6_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x6_train_sm = sm.add_constant(x6_train_sm)
# create a fitted model in one line
mlm6 = sm.OLS(y_train,x6_train_sm).fit()

# print the coefficients
mlm6.params
print(mlm6.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred6, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x6_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x6_train.values, i) 
                          for i in range(len(x6_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred6, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred6)
r_squared = r2_score(y_test, y_pred6)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


x6_train = pd.DataFrame(x6_train)
y_train = pd.DataFrame(y_train)


# In[ ]:


df2 = pd.concat([x6_train,y_train],axis = 1)


# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(df2.corr(),annot = True)
plt.show()


# #### `MODEL-8`

# In[ ]:


x7_train = x_train[['avgAnnCount', 'incidenceRate', 'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]
x7_test = x_test[['avgAnnCount', 'incidenceRate', 'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]


# 'avgDeathsPerYear', 'popEst2015' This varibles is removed because of less correlation with Target_death rate

# In[ ]:


lr7 = lr.fit(x7_train,y_train)


# In[ ]:


y_pred7 = lr7.predict(x7_test)


# In[ ]:


import statsmodels.api as sm
x7_train_sm = x7_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x7_train_sm = sm.add_constant(x7_train_sm)
# create a fitted model in one line
mlm7 = sm.OLS(y_train,x7_train_sm).fit()

# print the coefficients
mlm7.params
print(mlm7.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred7, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x7_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x7_train.values, i) 
                          for i in range(len(x7_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred7, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred7)
r_squared = r2_score(y_test, y_pred7)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


x7_train = pd.DataFrame(x7_train)
y_train = pd.DataFrame(y_train)


# In[ ]:


df3 = pd.concat([x7_train,y_train],axis = 1)


# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(df3.corr(),annot = True)
plt.show()


# #### `MODEL-9`

# In[ ]:


x8_train = x_train[['avgAnnCount', 'incidenceRate', 'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]
x8_test = x_test[['avgAnnCount', 'incidenceRate', 'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]


# 'PctPublicCoverage', This varibles is removed because of less correlation with Target_death rate

# In[ ]:


lr8 = lr.fit(x8_train,y_train)


# In[ ]:


y_pred8 = lr8.predict(x8_test)


# In[ ]:


import statsmodels.api as sm
x8_train_sm = x8_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x8_train_sm = sm.add_constant(x8_train_sm)
# create a fitted model in one line
mlm8 = sm.OLS(y_train,x8_train_sm).fit()

# print the coefficients
mlm8.params
print(mlm8.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred8, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x8_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x8_train.values, i) 
                          for i in range(len(x8_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred8, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred8)
r_squared = r2_score(y_test, y_pred8)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


x8_train = pd.DataFrame(x8_train)
y_train = pd.DataFrame(y_train)


# In[ ]:


df4 = pd.concat([x8_train,y_train],axis = 1)


# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(df4.corr(),annot = True)
plt.show()


# #### `MODEL-10`

# In[ ]:


x9_train = x_train[['avgAnnCount', 'incidenceRate', 'MedianAgeMale', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]
x9_test = x_test[['avgAnnCount', 'incidenceRate', 'MedianAgeMale', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 
                    'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctOtherRace', 
                    'PctMarriedHouseholds']]


# 'PercentMarried', This varibles is removed because of Higher VIF value

# In[ ]:


lr9 = lr.fit(x9_train,y_train)


# In[ ]:


y_pred9 = lr9.predict(x9_test)


# In[ ]:


import statsmodels.api as sm
x9_train_sm = x9_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x9_train_sm = sm.add_constant(x9_train_sm)
# create a fitted model in one line
mlm9 = sm.OLS(y_train,x9_train_sm).fit()

# print the coefficients
mlm9.params
print(mlm9.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred9, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x9_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x9_train.values, i) 
                          for i in range(len(x9_train.columns))] 
  
print(vif_data)


# In[ ]:


c = [i for i in range(1,611,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred9, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred9)
r_squared = r2_score(y_test, y_pred9)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


x9_train = pd.DataFrame(x9_train)
y_train = pd.DataFrame(y_train)


# In[ ]:


df5 = pd.concat([x9_train,y_train],axis = 1)


# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(df5.corr(),annot = True)
plt.show()


# In[ ]:


x9_train.columns


# In[ ]:


avgAnnCount = int(input('Average Annual Count:' ))
incidenceRate = int(input('Incident Rate:' ))
MedianAgeMale = int(input('Age of Male:' ))
PctHS18_24 = int(input('High School Diploma 18-24 in a country:' ))
PctHS25_Over = int(input('High School Diploma >25 in a country:' ))
PctBachDeg25_Over = int(input('Bac. Degree 18-24 in a country:' ))
PctEmployed16_Over = int(input('Employed over 16 in a country:' ))
PctPrivateCoverage = int(input('Private Health coverage in a country:' ))
PctEmpPrivCoverage = int(input('Employee - Private Health coverage in a country:' ))
PctOtherRace = int(input('Residents - not White, Black, or Asian:' ))
PctMarriedHouseholds = int(input('Married Households:' ))


# In[ ]:


Intercept = mlm9.params[0]


# In[ ]:


Coefficient = mlm9.params[1]*avgAnnCount + mlm9.params[2]*incidenceRate + mlm9.params[3]*MedianAgeMale + mlm9.params[4]*PctHS18_24 + mlm9.params[5]*PctHS25_Over + mlm9.params[6]*PctBachDeg25_Over + mlm9.params[7]*PctEmployed16_Over + mlm9.params[8]*PctPrivateCoverage +  mlm9.params[9]*PctEmpPrivCoverage + mlm9.params[10]*PctOtherRace + mlm9.params[11]*PctMarriedHouseholds


# In[ ]:


y_Predicted = Intercept + Coefficient  # y = a + bx  Equation.
print('Predicted Death Rate for unkown values:', y_Predicted)


# ### Found that the Model 10 is best fitted and having the accuracy of R2 value is 50%. with 11 number of variables.

# In[ ]:





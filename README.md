# EX-6-Feature-Transformation
Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation is a technique by which we can boost our model performance.

# ALGORITHM
STEP 1 Read the given Data

STEP 2 Get the information about the data

STEP 3 Perform function transformation

STEP 4 Analyse Power transformation

STEP 5 Analyse Quantile transformation

# CODE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("/content/Data_to_Transform.csv")
df.head()   

Function Transformation

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()


sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()


sm.qqplot(df. ModeratePositiveSkew, fit=True, line='45')
plt.show()


sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

LOG Transformation


df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)

sm.qqplot(df.HighlyPositiveSkew, fit=True, line='45')
plt.show()

RECIPROCAL Transformation


df['HighlyPositiveSkew']=1/df.HighlyPositiveSkew
sm.qqplot(df.HighlyPositiveSkew, fit=True, line='45')
plt.show()

Square root Tranaformation

df[ 'HighlyPositiveSkew']
np.sqrt(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

Power Transformation

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer ("yeo-johnson")
df['ModerateNegativeSkew_1']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df[ 'ModerateNegativeSkew_1'],line='45')
plt.show()

Quantile Transformation

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer (output_distribution = 'normal')
df['ModerateNegativeSkew_3']=pd.DataFrame(qt.fit_transform(df[[ 'ModerateNegativeSkew']]))
sm.qqplot(df[ 'ModerateNegativeSkew_3'], line='45')
plt.show()

# RESULT
Thus we have performed Feature Transformation is a techniques by which we can boost our model performance.



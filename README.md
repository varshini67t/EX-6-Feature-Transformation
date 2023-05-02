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
# OUTPUT

![image](https://user-images.githubusercontent.com/107982953/235741473-5eb43516-01be-4238-8f01-574e10cd163d.png)
![image](https://user-images.githubusercontent.com/107982953/235741589-5374de6b-cfbb-4d86-b028-a3e2b35e0552.png)
![image](https://user-images.githubusercontent.com/107982953/235741646-f51a030c-1301-4cbf-9260-0309a48067dd.png)
![image](https://user-images.githubusercontent.com/107982953/235741708-3a67fe73-98d9-40d1-873c-496d0642ce67.png)
![image](https://user-images.githubusercontent.com/107982953/235741779-65af1c88-6957-4ea7-b6cf-647c8f7c9b4d.png)
![image](https://user-images.githubusercontent.com/107982953/235741844-49d357cb-da7c-4727-b886-cdd7f534cf7b.png)
![image](https://user-images.githubusercontent.com/107982953/235741924-8b8bcf44-1590-401c-a5f3-18d738f26dd8.png)
![image](https://user-images.githubusercontent.com/107982953/235741999-fab1f162-ef89-4174-b792-e6f32c146923.png)
![image](https://user-images.githubusercontent.com/107982953/235742067-4e4b1efd-504b-46c4-b80b-9d13ecf398be.png)
![image](https://user-images.githubusercontent.com/107982953/235742145-7b3e9232-4df0-4032-8a7e-a90bd1654133.png)

# RESULT
Thus we have performed Feature Transformation is a techniques by which we can boost our model performance.



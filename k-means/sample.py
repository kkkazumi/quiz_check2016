import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
 
cust_df = pd.read_csv("http://pythondatascience.plavox.info/wp-content/uploads/2016/05/Wholesale_customers_data.csv")
print(cust_df)
 
del(cust_df['Channel'])
del(cust_df['Region'])
cust_df

cust_array = np.array([cust_df['Fresh'].tolist(),
                       cust_df['Milk'].tolist(),
                       cust_df['Grocery'].tolist(),
                       cust_df['Frozen'].tolist(),
                       cust_df['Milk'].tolist(),
                       cust_df['Detergents_Paper'].tolist(),
                       cust_df['Delicassen'].tolist()
                       ], np.int32)
 
cust_array = cust_array.T
 
pred = KMeans(n_clusters=5).fit_predict(cust_array)
print(pred)

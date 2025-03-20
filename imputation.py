import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

data_columns = ['ID', 'Name', 'Age', 'Height', 'Weight', 'Eye_Color']
data = [[1,'Harry',23,172,65,None],
        [2,'Sally',45,None,75,'Green'], 
        [3,'Sabrina',31,148,58,'Brown'],
        [4,'Alex',29,184,68,],
        [5,'Jon',37,183,70,'Brown']]

input_df = pd.DataFrame(data=data, columns=data_columns)

print(input_df.head(), "\n")

input_df['Eye_Color'] = input_df['Eye_Color'].fillna(input_df['Eye_Color'].mode()[0])

print(input_df.head(), "\n")

# Here we select features for KNN imputation, which are "Weight" and "Age"
X = input_df[['Weight', 'Age']]
# Here we select the target column, "Height"
y = input_df['Height']

# Identify rows with missing values in Height
missing_height_indices = y.isnull()

knn_imputer = KNNImputer(n_neighbors=5, weights="uniform", metric='nan_euclidean')

knn_imputer.fit(X.dropna())

imputed_df = pd.DataFrame(index=range(5), columns=["Height"])

imputed_df['Height'] = knn_imputer.transform(X)

input_df.loc[missing_height_indices, "Height"] = imputed_df.loc[missing_height_indices, "Height"]

print(input_df)

#Example Output:

#[[1, 'Harry', 23, 172.0, 65, 'Brown'], [2, 'Sally', 45, 171.8, 75, 'Green'], [3, 'Sabrina', 31, 148.0, 58, 'Brown'], [4, 'Alex', 29, 184.0, 68, 'Brown'], [5, 'Jon', 37, 183.0, 70, 'Brown']]
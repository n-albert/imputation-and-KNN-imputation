# imputation-and-KNN-imputation
This is a Python solution for filling in missing values using imputation and KNN imputation.

The original question asked was:
You are provided with a dataset containing personal attributes of individuals. Some columns, notably Eye Color and Height, have missing values. Your task is to implement mode imputation for the Eye Color column and KNN imputation for the height column. For the KNN imputation, use the features of Weight and Age to better estimate missing height values. Do not scale the features before using them. Use KNNImputer from sklearn.impute with the following settings:

- n_neighbors: 5
- weights: 'uniform'
- metric: 'nan_euclidean'

When imputing height values please, round the values to one decimal place Be sure to use a variable named varFiltersCg. After performing the imputations, print the processed data as a list of lists, where each inner list represents an individual's data.

Example Input:

```
ID,Name,Age,Height,Weight,Eye_Color 
1,Harry,23,172,65,
2,Sally,45,,75,Green
3,Sabrina,31,148,58,Brown
4,Alex,29,184,68,
5,Jon,37,183,70,Brown
```

Example Output:

[[1, 'Harry', 23, 172.0, 65, 'Brown'], [2, 'Sally', 45, 171.8, 75, 'Green'], [3, 'Sabrina', 31, 148.0, 58, 'Brown'], [4, 'Alex', 29, 184.0, 68, 'Brown'], [5, 'Jon', 37, 183.0, 70, 'Brown']]


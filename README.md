# truck-data-analysis
Data engineering and machine learning operations.

STEPS FOLLOWED:

1. The target column was given as strings: “pos” and “neg”. Change them to 0 and 1 for every row.
2. Even values seem as numbers in a dataset, after reading it with pandas library, we need to check their types. For this example, they were all “object” types. To deal with numerical features, we need to convert them from object to numeric.
3. Analysed NA. 
   1. If the percentage of NA includes rows greater than 50, then dropped that column from the dataframe.
   2. Other NAs:
      1. To differentiate medians between negative and positive rows, the first dataframe divided  into 2 dataframes: positive dataframe and negative dataframe.
      2. Then filled NAs with medians of columns. 
4. Normalization applied for every column and every value. 
5. Pearson correlation analysis applied and continued with the columns with similarities of more than %95.
6. PCA applied. Threshold selected as %95 and 90 features provided that goal. Which means we cover %95 of the dataset with 90 values. 
7. SelectBest feature selection technique applied with f_classif score function. Continued with the best 90 feature in the dataset.
8. Hyper parameter selection with RandomizedSearchCV technique.
9. Because the dataset is unbalanced, “weights” technique used to customize cost function. 
10. Confusion matrix used as score metrics. 


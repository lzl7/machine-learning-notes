# Decision Tree
It learns how to split the data into smaller subsets to predict the target

## Decision tree algorithms
ID3 and CART are the popular two algorithms. 

### ID3
Each node can have two or more splits/edges. It tries to maximize the information gain using the impurity criterion entropy.

ID3 can only used for classification problems.

### CART
Classification and Regression Trees (CART) only create two splits (i.e. binary tree). It uses the impurity criterion entropy to find the best numerical or categorical feature. It supports both numerical and categorical variables.

For classification, the criterion is Gini impurity or twoing criterion while for regression it use variance reduction using least squares (RMSE).

Further reading: [Comparative Study ID3, CART and C4.5 Decision Tree Algorithm: A Survey](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.685.4929&rep=rep1&type=pdf)

## Criterion
Impurity formulas: 
![Impurity formulas](https://miro.medium.com/max/5019/1*WDR11Z14B0YgNiKeksxlEQ.png) from srnghn in medium.



## Reference
- https://medium.com/@srnghn/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3
- Further reading: https://medium.com/greyatom/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb
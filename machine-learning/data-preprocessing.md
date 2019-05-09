Look into the data and have a deep full understanding the features, the data distribution is very important for data scientist. And it is also the first step to hand on the project.

## pandas.transpose()

This is method is basically transpose the row and col. It is really helpful for looking the data when the columns/features number is large.

You could use the `pandas.head()` or `pandas.tail()` to pick some data and then transpose them. Then all the feature will be represented as the rows and the columns is the data for different entries.

There is also another way to limit the output size for the column and row. `pandas.option_context()` could allow to set the max rows and columns. Example: `with option_context('display.max_rows', 10, 'display.max_columns', 5):`. Btw, by using the `with` limits the scope of the setting and avoid impacing others.

## pandas.to_feather() & pandas.read_feather()

Output the binary feather-format of the DataFrame which contains all the pre-processed data. In next time, we just need to reload the data from this middle output and continue to experiment the training. This solution could help us to avoid doing the duplicated preprocessing process. 
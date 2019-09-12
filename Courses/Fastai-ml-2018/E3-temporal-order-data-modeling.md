# Validation set
The random shuffling sample approach to select the validation set might not working in this scenario. Because the validation set might contain the samples from the future.

If the validation set is good, say takes the most recent records away from the training set to do evaluate. If the model performs well with OOB score, but bad on the validation set. Then it means that the model does a bad job on predict the future instead of overfitting. The suggestion would be: revisit the training data, and start with the recent records instead of the whole history dataset.

## Validation set vs. Test set?
Once you have a working well model on the validation set, don't go to test directly in the test set. Since the test set samples is much farther. You could combine the training set and validation set and retrain the model again with exact approach in the training. As this time, you don't have validation set any more, so you need to make sure to do exact same way as previoud did. The key thing for this approach is to make sure that the validation set is a truely representative of the test set. How?
- Build several (say 5) models on training set. Try to have them vary, and they don't need to be very accurate.
- Run the models on both the validation set and test set
- Draw the chat based on the scores from both sets, and check whether they are linear related

*What if the test on validation set and test set don't get straight-line relationship?*
You should not change test set, but adjust the validation set.

*How to train the 5 models?*
Usually, the models should have some variety, say one trained on whole training set, one on last two months, one on last two weeks, same day range one month earlier etc.

***The whole point is to make the validation set representative to the test set, which means gains on the evaliation set should also reflect (improve) in test set at some extent.***

# Test set
When build test set for production, we need to understand (including but not limited to):
- What is the actual customers use the model?
- How much time it takes between building model to running in production?
- How often it needs to refresh the model?

# Suggestions for temporal data
A simple way is try to avoid using time variables as predicators if something else could be used to show better or stronger relationship that could used in the future predictions

## How to figure out other variables instead of time?
1. Figure out what is difference between validation set and training set. *How?*
   - Create a random forest, and the dependent variable is "is it in the validation set" (`is_valid`)
     - Create a new data frame with training and validation set together
     - Create a new column called `is_valid`, with training set item as 0 validation's as 1
     - If yes, use the feature importance to understand the top N (time related) columns in both training and validation set.
       - If they are clearly time dependent, drop them and retrain. And see whether it can still predict `is_valid` and continue the process to understand the features.
       - Please note that, you might still want them in the random forst if they are important. If not, drop them if there are some other none-time depdendent variable works well. *The whole point is try to remove the time dependent but still keep the accuracy.*
   - *How to use the random forest?* 
     - If it is not time dependent, it shouldn't be possible to figure out if something `is_valid` or not
   - *Trick in Kaggle*: 
     - How to test/understand whether the test set is random sample or not?
       - Put the training and test set together and create a new column `is_test` and see whether you can predict it
     - ***Why need to understand it**?*
       - If you can, you don't have a random sample, it means you need to figure out how to create a valiation set
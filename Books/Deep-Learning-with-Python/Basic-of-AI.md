# Relationship between: Artificial Intelligence, Machine Learning, Deep Learning

## Basic Definitions
**Artificial Intelligence (AI)** refers to the effort to automate intellectual tasks normally performed by humans.

**Machine Learning**: is the algorithms to learn without explicitly programmed. 

**Deep  Learning**: is a special subfield of machine learning, it emphases on the successive layers of increasing learning.

Hence, AI includes the machine learning as well as deep learning, and also other approaches that even without any learning like handcraft programmed rules.

## Evolutions

### Artificial Intelligence Early Age
- AI born in 1950s
- *Symbol AI* (1950s-1980s, No Learning): handcraft rules
    - Expert Systems: 1980s

Symbol AI is suitable to solve well-defined logical problems. However, it barely helps on the complex fuzzy problems. So, the new approach - *Machine learning* comes out.

### Machine Learning (ML)

Machine Learning started flourish in 1990s, it tries to achieve:
- Beyond knowing perform what programmer order, but learn to perform specified tasks.
- Automatically learn the rules from the data instead of only receiving handcraft rules from programmer

***1. What is the difference between ML and classical programming?***

ML is *trained* rather than *explictly programmed*. The comparison between classic programming and ML:
- Classical programming:
    - Input: data and rules
    - Output: *answers*
- Machine learning:
    - Input: data and answers
    - Output: *rules*

***2. What does machine learning do?***

Machine learning is learning the representation of data. There are three key things for machine learning:
- **Input data points**: the input resource used for training 
- **Expected output examples**: the labelled data as the ground truth for training prediction 
- **Approach to measure the algorithm performance (how good)**: measure the difference between the prediction and the expected output, this measurement provides the feedback signal to algorithm to adjust the training. The adjustment step is called *learning*.

The central problem in machine learning is all about representation, i.e. *meaningfully transform data*. Representation is a diffferent way to present or encode the input data.

***Learning*** is an automatic search process for better representation in machine learning. Machine learning algorithms aren't usually creative finding the transformations, but merely search through the predefined set of operations (called *hyphothesis space*).

> **Machine learning is, technically: searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.**

***3. History of Machine learning***

1. **Probabilitic Modeling**, apply statistic to data analysis
    - Naive Bayes algorithm
2. **Early Neural Networks**
    - Toy forms as early as 1950s
    - Efficient leaning changed in 1980s by backpropagration algorithm using gradient-descent optimization
    - First application LeNet by Yann LeCun, recognizing ZIP code 
3. **Kernel Methods**
    - Kernel methods are group of classification algorithms (prefer by 2010)
        - Support Vector Machine (SVM)
            - Goal: finding good decision boundaries to split two categories
            - Two steps:
                - Map to high-dimensional representation where the decision boundary can expressed as a hyperplane
                - Maximizing the margin: A good decision boundary is computed by trying maximize the distance
            - Cons
                - Hard (computationally) in practice
                - *Kernel trick* to resolve it by kernel function (typically crafted by hand rather than learned from data)
                    - Map any two points to the distance between them in the new space, by passing the computation of new representation
                    - Then SVM only the separaton hyperplane is learned
                - Hard to scale to large dataset
4. **Decision Trees, Random Forests, and Gradient Boosting Machines**
    - Decision tree: flowchart-like strcture (2000s)
    - Random Forest: build a large number of decision trees and then ensembiling their outputs (*bagging*)
    - Gradient Boosting Machines: iteratively train new models that specialize in addressing the weak points of previous models, aka. gradient boosting method.
5. **Back to Neural Networks**, after 2010
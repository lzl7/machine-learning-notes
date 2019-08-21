# Machine learning solution modeling

> Vast majority of datasets can be best modeled with just two methods:
> - ***Ensembles of decision trees*** (i.e. Random Forests and Gradient Boosting Machines), mainly for **structured data** (such as you might find in a database table at most companies)
> - ***Multi-layered neural networks learnt with SGD*** (i.e. shallow and/or deep learning), mainly for **unstructured data** (such as audio, vision, and natural language)

# Before modeling
## 1. Try to understand the data first
It doesn't mean that you have to study the data very carefully and try to figure out the pattern based on that (which might lead to the overfitting), but understand how the format looks like, what is the data type, etc.

> It's **important to look at your data**, to make sure you understand the format, how it's stored, what type of values it holds, etc. Even if you've read descriptions about your data, the actual data may not be what you expect.

## 2. Understand the measuremnent metrics
> Generally, selecting the metric(s) is an important part of the project setup.
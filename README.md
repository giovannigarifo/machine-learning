# Machine Learning

This repository contains the homeworks for the cours [Machine Learning and Artificial Intelligence](https://didattica.polito.it/pls/portal30/gap.pkg_guide.viewGap?p_cod_ins=02SQING&p_a_acc=2019&p_header=S) of the master degree in Computer Engineering of Politecnico di Torino.

The topics of the homeworks are:

* Principal Component Analysis
* Support vector Machines
* Convolutional Neural networks


# Introduction to Machine Learning

A clarification about how ML and AI are related, the two common definitions are:

* **AI**: theory and algorithms that enable computers to mimic human intelligence
  
* **ML**: a subset of the AI that uses statistical techniques that enable machines to improve at tasks with experience. 
  

In other words, key point of ML is to program computers to use example data or past experience to build a "knowledge" that will allow them to perform some basic tasks better then the last time they have done it.

This can be formalized as the **learning problem**:

* a computer program is said to learn from ad *experience E*
* with respect to *class of tasks T* and *performance measure P*
* IF its performance at task *T*, as measured by *P*, improves by doing experience *E*

ML in based on data, and to program with those data. 

Basic strategy of creating a ML model is to collect data as (x,y) pairs, where **"x" is the data**, i.e. a picture of a phone, and **"y" is the label**, i.e. the label that tells that in that picture, there's a phone.

Then, using those data, we can:

* **estimate** a function *f* such that **f(x)=y**, that allow us to **classify** other data (supervised learning).
* or if we don't have labeled data to train the model, or if we don't trust the labels that we have, we can detect patterns in data and group them in **clusters** (unsupervised learning).


Two major branches of ML tasks exists:

## Supervised learning

The machine learning task of learning a function that maps an input "x" (data) to ad output "y" (label), based on example input-output labeled pairs that are used to train the model. It allow to infer a function from a set of labeled training data. An example of applications of such tasks are the classification or regression algorithms. 

Some example of supervised learning model:

* Binary classifier: given x, find y in {-1,1}
* Multicategory classifier: given x, find y in {1,2,3,...,k}

    often we use along with the *f* another function, called the **loss function l(y,f(x))** that can be used to check if the algorithm is behaving like we want, it has to be designed along with the f. It can be used as a benchmark of the algorithm.

* Regression classifier: given x find y in *R*

    
Two kinds of supervised learning models exists:
    
1. **Discriminative learning algorithms**: using directly the *conditional probability distribution* **p(y|x)**, they allow to model the dependence of an unobserved variable "y" on an observed variable "x". i.e.: logistic regression, gradient descent. Offers better convergence, simpler solutions. Very good with complicated data, such as images or texts. Works well if a lot of data are available, converging to a solution in most of the cases. More simple because the model only cares abour estimating the conditional probabilities and finding the decision function that will be used to classify the data.

2. **Generative learning algorithms**: based on the *joint probability distribution* **p(x,y)** and by applying the joint probability distribution p(y|x), they allow to generate likely (x,y) pairs. Good for missing variables, offers better diagnostics, but are more demanding. Easy to add prior knowledge about data.
   
    
## Unsupervised learning

The algorithm learns from data that has not been labeled, classified or categorized, searching for commonalities in the data, like an hunt for relationships. Such commonalities can then be used to analyze a new piece of data, with better knowledge. An example of applications of such tasks are clustering algorithms.


## Essential statistics concepts for ML understanding

### **Probability**

Given an experiment, Ω as the space of all the available results of that experiment, we define P(A) as the probability that the result of the experiment will be A. 

But we are using the "probability" to define the probability itself. There's another definition, called the assiomatic definition, based on the three **Kolmogorov axioms**:

* P(A)>=0, for each A
* P(Ω) = 1
* additivity of disjoint results: P(Ω) = sum of all P(A*i*), for each *i*.

Some consequences are that:

* P(0) = 0
* P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
* P(negated A) = 1 - P(A)

### **Random variables and probability distributions**

A function P(X) of the result of a randomized experiment. 

    X : Ω -> R

They are used to associate a real number to each element of the Ω set, they can be both discrete or continuous. Now that we have a mathematical function, we can study it's distribution.

* Discrete distributions
  
* Continuous distributions

    For a continuous distribution we can define the **cumulative distribution function (CDF)**: 

        Fx(z) = P(x <= z)

    If it's absolutely continuous, the first derivative of the Fx(z) is called **probability density function (PDF)**, noted as p(x).

Quantitative measures for the probability distribution are called moments, some moments are:

* **Mean** E(X), the average value or expected value of the probability distribution.
* **Variance** V(X), how much the values of the probability distribution differs from it's mean.


### **Joint distribution**

We can extend the above concepts from one dimension to n dimensions, where a dimension is a random variable. The joint probability represents the probability that both events occurs for each couple (or n-uple) of they're possible results. Notation: 

    P(X∩Y) = P(X=x ∩ Y=y) = P(x,y)


### **Conditional probability**

The probability that X occurs, given that Y has occured.

    P(X|Y) = P(X∩Y)/P(Y) = P(X,Y)/P(Y)

### **Independence**

Two events are indipendent if they are not related by any means, so they don't contain information about each other.

    P(X,Y) = P(X)*P(Y) => P(X|Y) = P(X)


### **Conditional independence**

Knowing the result of one of the experiment, makes the other two independent.

    P(X,Y|Z) = P(X|Z)*P(Y|Z)

i.e.: shoe size and reading skills given age.

### **Chain rule and Bayes rule**

**Chain rule**:

    P(X,Y) = P(X|Y)*P(Y) = P(Y|X)*P(X)

**Bayes rule**:

    P(X|Y) = P(Y|X)*P(X)/P(Y)


**Naive Bayes assumption**, given X1 and X2 conditionally independent

    P(X1,X2|Y) = P(X1|Y)*P(X2|Y)


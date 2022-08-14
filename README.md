# Spam Filter Using Naive Bayes

***PS**: this file may **not** be displayed at its best when viewed from a phone, due to the formatting of some math formulas. It is suggested to view it from a computer.*

### Table of Contents
* [Introduction](#introduction)
* [Folder Structure](#folder-structure)
* [Some Necessary Math Review](#some-necessary-math-review)
* [What is Naive Bayes?](#what-is-naive-bayes)
* [Project Procedure](#project-procedure)
    * [Example](#example)
* [Limitations of the Naive Bayes model](#limitations-of-the-naive-bayes-model)
* [Results](#results)
* [Curiosities](#curiosities)

## Introduction

We know that spam is usually related to emails. But spam can occur also in text messages! The following project is a spam filter that classifies text messages as spam or non-spam using a Naive Bayes algorithm.

This README is more of an introduction to the model, with a short summary of the results of the project at the end. Throughout the file and the project I will refer to myself as "we". Please, bear in mind that there's no "we", it's just me.

## Folder Structure

* **data** folder: contains the dataset and its dictionary
* **src** folder: contains the code of the project

## Some Necessary Math Review

Let's do some review of the crucial topics needed to understand what Naive Bayes is.

* **Conditional probability**: Given two events $A$ and $B$ of a sample space $\Omega$, with $P(B)>0$, we define the probability of $A$ given $B$ as 
    $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
* **Law of Total Probability (LTT)**: If  $A$ is an event of a sample space $\Omega$, and $B_n$ is a partition of $\Omega$, then
    $$P(A) = \sum_{i = 1}^{n}P(A \cap B_i) = \sum_{i = 1}^{n}P(A|B_i)P(B_i) $$
* **Bayes' Theorem**: If $A$ and $B$ are two events of a sample space $\Omega$, with $P(A)>0$, then
    $$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$$ In Bayesian terms we can also state the theorem as
    $$posterior = \frac{likelihood \cdot prior}{evidence}$$ Using LTT we can also restate the theorem in another form:
    $$P(B_i|A) = \frac{P(A|B_i)P(B_i)}{P(A)} = \frac{P(A|B_i)P(B_i)}{\sum_{j\in \{1,\dots,n\} } P(A|B_j)P(B_j)}$$
* **Conditional Independence**: Let $A$, $B$ and $C$ be three events of a sample space $\Omega$. We say that $A$ and $B$ are conditionally independent given $C$, with $P(C)>0$, if
    $$P(A|B,C) = P(A|C)$$ An alternative characterization of conditional independence is the following:
    $$P(A,B|C) = P(A|C)P(B|C)$$ More generally, if some events $A_1, A_2, \dots, A_n$ are conditionally independent given $B$, then
    $$P(A_1, A_2, \dots, A_n |B) = \prod_{i = 1}^{n}P(A_i|B)$$ and
    $$P(A_i|A_1, \dots , A_{i-1}, A_{i+1} \dots, A_n, B) = P(A_i|B)$$   
    This holds true for random variables as well. I used events just to remain in the realm of pure probability.

## What is Naive Bayes?

Naive Bayes is a probabilistic model based on Bayes' theorem. It is used in machine learning for classification problems. 

In a typical classification problem, we are given some observations $X_1, X_2, \dots, X_n$ and a class variable $C_k$, where $k$ represents a possible outcome of the classification.
For demonstration purposes assume that the outcome is binary, each outcome denoted by $+$ and $-$. That is $K = \{+, - \}$.

The Naive Bayes classifier assigns a class label $k\in K$ to our observations. Technically speaking, it assigns a value to
$$P(C_k|X_1, \dots, X_n)\ \forall k\in K$$
We say that the input is classified as the class $+$ if
$$P(C_{+}|X_1,\dots ,X_n) > P(C_{-}|X_1,\dots ,X_n)$$ The opposite holds for the $-$ class as well.

That's where "Bayes" comes into play. From Bayes' theorem, we know that 
$$P(C_k|X_1, \dots, X_n) = \frac{P(X_1, \dots, X_n|C_k)P(C_k)}{P(X_1, \dots, X_n)}$$

We can notice that our $evidence$, i.e. the denominator, does not depend on the class variable $C_k$ and it is always constant. So we can drop the denominator and introduce a proportionality, instead of keeping an equality. This change will not affect at all the the ability to classify correctly the class variable. The model arrives at the correct classification as long as the correct class is more probable than any other class; hence class probabilities do not have to be estimated very well.
So we are left with 
$$P(C_k|X_1, \dots, X_n) \propto P(X_1, \dots, X_n|C_k)P(C_k)$$

That's where the "Naive" assumption comes into play: we assume that the observations $X_1, \dots, X_n$ are conditionally independent given the class variable $C_k$.
Hence, $$P(X_1, \dots, X_n|C_k)P(C_k) = P(C_k) \prod_{i = 1}^{n}P(X_i|C_k)$$

We'll see later on why this assumption is called "naive", but with some thought you might guess the reason why.

It turns out the the final probabilistic model of Naive Bayes is the following:
$$P(C_k|X_1, \dots, X_n) \propto P(C_k) \prod_{i = 1}^{n}P(X_i|C_k)$$

If you feel confused, go on reading. I will show you an example that will hopefully clear your mind. In addition do not forget to check the code of the project!

## Project Procedure

It is fundamental to know that there are different versions of a Naive Bayes model: Bernoulli, Multinomial and Gaussian. Explaining their differences is out of my scope. We will use Multinomial Naive Bayes (MNB) for the project. In a MNB model, the observations $X_1, \dots, X_n$ each represent the number of times the event $i\in\{1,\dots ,n\}$ has occurred.

In our case, the observations $X_1, \dots, X_n$ are the number of occurrences of a specific word in a sentence. Our classification variable $C_k$ is represents whether a given message is "spam" or "non-spam". So we will try to solve a binary classification problem.

We will use a [dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) with more than 5000 messages that have already been classified as spam or non-spam.

We will first calculate the probability of the words appearing in a spam and non-spam message and the probability of a message being spam or non spam. We will also keep a count of the total number of words that appear in the whole set, the so-called "vocabulary".
From there we will be able to compute all the necessary probabilities to classify a message.

80\% of the dataset will used for training, while the remaining 20\% will be our test set.

### Example

Suppose we are given the following table, with given labels, and we want to understand whether the message "secret place secret secret" is spam or not.

| **LABEL** | **MESSAGE**                |
|:---------:|:--------------------------:|
| non-spam  | secret party at my place   |
| spam      | secret money secret secret |
| spam      | money secret place         |
| non-spam  | you know the secret        |
| ?         | secret place secret secret |

First we calculate the probability of a message being spam and non-spam
$$P(Spam)=\frac{1}{2}$$ $$P(Non\ Spam)=\frac{1}{2}$$ Then we construct a vocabulary $V$ with all the words. We get that
$$V = secret, party, at, my, place, money, you, know, the$$
After this, we calculate the frequency-based probabilities of each word in the dictionary. For example
$$P(secret|Spam) = \frac{4}{7}$$
$$P(secret|Non\ Spam) = \frac{2}{9}$$
$$P(place|Spam) = \frac{1}{7}$$
$$P(place|Non\ Spam) = \frac{1}{9}$$
Using our Naive Bayes approximation we have that the probability of the message being spam is:
$$P(Spam|secret, place, secret, secret) = $$
$$= P(secret|Spam)^3\cdot P(place|Spam)\cdot P(Spam) = \left(\frac{4}{7}\right)^3\cdot \frac{1}{7}\cdot\frac{1}{2} = 0.01333$$
The probability of the message not being spam is:
$$P(Non\ Spam|secret, place, secret, secret) = $$
$$= P(secret|Non\ Spam)^3\cdot P(place|Non\ Spam)\cdot P(Non\ Spam) = \left(\frac{2}{9}\right)^3\cdot \frac{1}{9}\cdot\frac{1}{2} = 0.0006097$$
We now see that $0.01333 > 0.0006097$, so the message is classified as spam.
We can notice that the final probabilities do not sum up to 1, as they normally should. That's because we're not dividing by the joint probability of the features, as we should do due to Bayes' Theorem. Nonetheless, the proportions are the same, so the result is still valid.

We will proceed with a similar method in our project, of course on a larger data set.

## Limitations of the Naive Bayes model

Since the probability is proportional to the number of occurrences of a feature (in this case a specific word), if a word never appears for a specific class (spam or non-spam) during the training set, than it's probability of appearing in that class will always be 0. 

This is not ideal. Let's look at an example.


| **LABEL** | **MESSAGE**                |
|:---------:|:--------------------------:|
| non-spam  | secret party at my place   |
| spam      | secret money secret secret |
| spam      | money secret place         |
| non-spam  | you know the secret        |
| ?         | secret code to unlock the money|

We can see that "code" and "unlock" are not in the vocabulary, "money" appears only in spam messages, "the" only in non-spam.
Again, calculating the probabilities, we have

$$P(Spam) = \frac{1}{2}$$

$$P(Non\ Spam) = \frac{1}{2}$$

$$P(the|Spam) = \frac{0}{7}$$

$$P(money|Non\ Spam) = \frac{0}{9}$$

This is enough for us to show that using Naive Bayes we get
$$P(Spam|secret, code, to, unlock, the, money) = 0$$
$$P(Non\ Spam|secret, code, to, unlock, the, money) = 0$$

This is problematic: both values yield 0. To solve this issue, we do what is called **additive smoothing**. This is a correction we do to our data, such data no value has a frequency of value 0. We do so by adding a smoothing parameter $\alpha$.

Let's try to understand a bit more. Normally, we have that the probability of event $i$ is $p_i = \frac{x_i}{N}$, where $x_i$ is the frequency of element $i$, and $N$ is the cardinality of our sample space. Our smoothed probability is defined as
$$p_i = \frac{x_i + \alpha}{N + \alpha \cdot d}$$
In this case $d$ would represent the dimension of the vocabulary, so $d = 9$ in our example set.

We will let $\alpha = 1$. When the parameter takes value one, we also call the procedure Laplace smoothing.

As a side-effect, though, this smoothing increases the probability of those elements with 0 probability, but it doesn't change the probability of non-zero events. To avoid this issue, we apply Laplace soothing to all observations in the dataset. In other words, we increase the frequency of all words by $\alpha$.

Let's look at how the probabilities change, after applying $\alpha = 1$
$$P(the|Spam) = \frac{0 + 1}{7 + 1\cdot 9} = \frac{1}{16}$$
$$P(money|Non\ Spam) = \frac{0 + 1}{9 + 1\cdot 9} = \frac{1}{18}$$
$$P(secret|Spam) = \frac{4+1}{7+9} = \frac{5}{16}$$
$$P(secret|Non\ Spam) = \frac{2 + 1}{9 + 9} = \frac{1}{6}$$
$$P(the|Non\ Spam) = \frac{1 + 1}{9 + 9} = \frac{1}{9}$$
$$P(money|Spam) = \frac{4+1}{7+9} = \frac{5}{16}$$
So we get that
$$P(Spam|secret, code, to, unlock, the, money) =$$
$$= P(secret|Spam)P(code|Spam)P(to|Spam)P(unlock|Spam)P(the|Spam)P(money|Spam)P(Spam) = 0.001831$$
$$P(Non\ Spam|secret, code, to, unlock, the, money) =$$
$$= P(secret|Non\ Spam)P(code|Non\ Spam)P(to|Non\ Spam)\cdot$$
$$\cdot P(unlock|Non\ Spam)P(the|Non\ Spam)P(money|Non\ Spam)P(Non\ Spam) = 0.000514$$

So, given the updated probability values, we can now classify the message as "spam". Note that we obviously didn't calculate probabilities for the missing words, such as "code" or "unlock".

In addition, a limitation of the Naive Bayes model is the naive assumption itself. In fact, it is called "Naive", because it is unlikely that the observations are independent in the real-world. For instance, we know that "I like pizza a lot", and "pizza a like I lot" are two different sentences because order matters. But the algorithm treats the two sentences as the same, due to the conditional independence assumption.

## Results
We ended up building a model with almost 99% accuracy. Some improvements can be done. See code for more on this.

## Curiosities
* Despite the strong assumption that we make on independence, Naive Bayes still performs fairly well in tasks like document classification and spam filtering. You can find a deep explation in [this paper](https://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf).
* I copy the extract from an [article](https://serokell.io/blog/naive-bayes-classifiers) I found online on Bayesian poisoning: `Bayesian poisoning is a technique used by email spammers to try to reduce the effectiveness of spam filters that use Bayes’ rule. They hope to increase the rate of false positives of the spam filter by turning previously innocent words into spam words in a Bayesian database. Adding words that were more likely to appear in non-spam emails is effective against a naive Bayesian filter and allows spam to slip through. However, retraining the filter effectively prevents all types of attacks. That is why Naive Bayes is still being used for spam detection, along with certain heuristics such as blacklist`
* The true applications of Naive Bayes in spam filtering can reach a much higher level of complexity!


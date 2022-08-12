# Spam Filter Using Naive Bayes

## Introduction

## Some Necessary Math Review

Let's do some review of the crucial topics needed to understand what Naive Bayes is.

* **Conditional probability**: Given two events $A$ and $B$ of a sample space $\Omega$, with $P(B)>0$, we define the probability of $A$ given $B$ as 
    $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
* **Law of Total Probability (LTT)**: If  $A$ is an event of a sample space $\Omega$, and $B_n$ is a partition of $\Omega$, then
    $$P(A) = \sum_{i = 1}^{n}P(A \cap B_i) = \sum_{i = 1}^{n}P(A|B_i)P(B_i) $$
* **Bayes' Theorem**: If $A$ and $B$ are two events of a sample space $\Omega$, with $P(B)>0$, then
    $$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$$ In Bayesian terms we can also state the theorem as
    $$posterior = \frac{likelihood \cdot prior}{evidence}$$ Using LTT we can also restate the theorem in another form:
    $$P(B_i|A) = \frac{P(A|B_i)P(B_i)}{P(A)} = \frac{P(A|B_i)P(B_i)}{ \sum_{j = 1}^{n} P(A|B_j)P(B_j)}$$
* **Conditional Independence**: Let $A$, $B$ and $C$ be three events of a sample space $\Omega$. We say that $A$ and $B$ are conditionally independent given $C$, with $P(C)>0$ if
    $$P(A|B,C) = P(A|C)$$ An alternative characterization of conditional independence is the following:
    $$P(A,B|C) = P(A|C)P(B|C)$$ More generally, if some events $A_1, A_2, \dots, A_n$ are conditionally independent given $B$, then
    $$P(A_1, A_2, \dots, A_n |B) = \prod_{i = 1}^{n}P(A_i|B)$$ and
    $$P(A_i|A_1, \dots A_{i-1}, A_{i+1} \dots, A_n, B) = P(A_i|B)$$   
    This holds true for random variables as well. I used events just to remain in the realm of pure probability.

## What is Naive Bayes?

Naive Bayes is a probabilistic model based on Bayes' theorem. It is used in ML for classification problems. 

In a typical classification problem, we are given some observations $X_1, X_2, \dots, X_n$ and a class variable $C_k$, where $k$ represents the possible outcome of the classification.
In this case we assume that the outcome is binary, each outcome denoted by $+$ and $-$. That is $K = \{+, - \}$.

The Naive Bayes classifier assigns a class label $k\in K$ to our observations. Technically speaking, it assigns a value to
$$P(C_k|X_1, \dots, X_n)\ \forall k\in K$$
We say that the input is classified as the class $+$ if
$$P(C_{+}|(X_1,\dots ,X_n)) > P(C_{-}|(X_1,\dots ,X_n))$$ The opposite holds for the $-$ class as well.

That's where "Bayes" comes into play. Indeed we know that 
$$P(C_k|X_1, \dots, X_n) = \frac{P(X_1, \dots, X_n|C_k)P(C_k)}{P(X_1, \dots, X_n)}$$

We can notice that our $evidence$, i.e. the denominator, does not depends on the class variable. What we do in Naive Bayes is drop the denominator, since it is always constant, and introduce a proportionality, instead of keeping an equality. This change will not affect at all the the ability to classify correctly the variable. So we are left with 
$$P(C_k|X_1, \dots, X_n) \propto P(X_1, \dots, X_n|C_k)P(C_k)$$

That's where the "Naive" assumption comes into play: we assume that the observations $X_1, \dots, X_n$ are conditionally independent given the class variable $C_k$.
Hence, $$P(X_1, \dots, X_n|C_k)P(C_k) = P(C_k) \prod_{i = 1}^{n}P(X_i|C_k)$$

We call this "Naive" assumption, because it is very unlikely to have independence between all the observations that we make.
For example, when analyzing a sentence, the sentences "I like pizza" and "pizza like I" will be the same for the algorithm. We'll see later a more precise example.

It turns out the the final probabilistic model of Naive Bayes is the following:
$$P(C_k|X_1, \dots, X_n) \propto P(C_k) \prod_{i = 1}^{n}P(X_i|C_k)$$

## Project Procedure

There are different versions of a Naive Bayes model: Bernoulli, Multinomial and Gaussian. Explaining their differences is out of my scope.

Multinomial Naive Bayes (MNB) will be used for the project. In a MNB model, the observations $X_1, \dots, X_n$ each represent the number of times the event $i\in\{1,\dots ,n\}$ has occurred.

We will use a [dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) with more than 5000 messages that have already been classified as spam or non-spam.

In our case, the observations are the number of occurrences of words in a sentence. Our classification variable $C_k$ is represents whether the message is "spam" or "non-spam".

We will first calculate the probability of the words appearing in a spam and non-spam message and the probability of a message being spam or non spam. We will also keep a count of the total number of words that appear, the so called "vocabulary".
From there we will be able to compute all the necessary probabilities.

80\% of the dataset will used for training, while the remaining 20\% will be our test set.

### Example

Suppose we are given the following table, with given labels and we want to understand whether the message "secret" is spam or not.

**PUT TABLE**

First we calculate the probability of the message being spam and non-spam
$$P(Spam)=\frac{1}{2}$$$$P(Non\ Spam)=\frac{1}{2}$$
Then first we construct a vocabulary $V$ with all the words. We get that
$$V = \{"secret", "party", "at", "my", "place", "money", "you", "know", "the"\}$$
After this, we calculate the frequency-based probabilities of each word in the dictionary. For example
$$P("Secret"|Spam) = \frac{4}{7}$$
$$P("Secret"|Non\ Spam) = \frac{2}{9}$$
While
$$P("Place"|Spam) = \frac{1}{7}$$
$$P("Place"|Non\ Spam) = \frac{1}{9}$$
Using our Naive Bayes approximation we have that the probability of the message being spam is:
$$P(Spam|"Secret","Place","Secret","Secret") = $$
$$= P("Secret"|Spam)^3\cdot P("Place"|Spam)\cdot P(Spam) = \left(\frac{4}{7}\right)^3\cdot \frac{1}{7}\cdot\frac{1}{2} = 0.01333$$

The probability of the message not being spam is:
$$P(Non\ Spam|"Secret","Place","Secret","Secret") = $$
$$= P("Secret"|Non \Spam)^3\cdot P("Place"|Non\ Spam)\cdot P(Non\ Spam) = \left(\frac{2}{9}\right)^3\cdot \frac{1}{9}\cdot\frac{1}{2} = 0.0006097$$
We now see that $0.01333 > 0.0006097$, so the message is classified as spam.
We can notice that the probabilities do not sum up to 1, as they normally should, that's because we're not dividing by the joint probability of the features. Nonetheless, the proportions are the same, so the result is still valid.

We will proceed with a similar method in our project, of course on a larger data set.

## Limitations of the Naive Bayes model

Since the probability is proportional to the number of occurrences of a feature (in this case a specific word), if a word never appears for a specific class (spam or non-spam) during the training set, than it's probability of appearing in that class will always be 0. 

This is not ideal. Let's look at an example.

**PUT TABLE**

We can see that "code" and "unlock" are not in the vocabulary, "money" appears only in spam messages, "the" only in non-spam.
Again, calculating the probabilities

$$P(Spam) = \frac{1}{2}$$
$$P(Non\ Spam) = \frac{1}{2}$$
$$P("the"|Spam) = \frac{0}{7}$$
$$P("money"|Non\ Spam) = \frac{0}{9}$$

This is enough for us to show that using Naive Bayes we get
$$P(Spam|"secret", "code", "to", "unlock", "the", money) = 0$$
$$P(Non\ Spam|"secret", "code", "to", "unlock", "the", money) = 0$$

This is problematic: both values yield 0. To solve this issue, we do what is called additive soothing.

In addition, a limitation of the Naive Bayes model is the assumption itself. In fact, it called "Naive", because it is unlikely that the observations are independent in the real-world. For instance, we know that "I like pizza a lot", and "pizza a like I lot" are two different sentences. But from a probabilistic point of view, the algorithm treats the two sentences as the same, due to the conditional independence assumption.

## Learning Objectives

## Useful Links

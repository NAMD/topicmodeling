topicmodeling
=============

Library containing tools for topic modeling and related NLP tasks.

It brings together implementations from various authors, slightly modified by me as well as a new visualization tools
to help inspect the results.

I have also added a fair ammount of tests, mainly to guide my refactoring of
the code. Tests are still sparse, but will grow as the rest of the codebase sees more usage and refactoring.

Quick tutorial
--------------

###Online LDA


The sub-package onlineldavb is currently  the most used/tested.
Here is a quick example of its usage:
Assume you have a set of documents you want to extract the most representative topics from. 

The first thing you need is a vocabulary list for these, i.e., valid informative words you may want to use 
to describe topics. I generally use a spellchecker to find these plus a list of stopwords.
*NLTK* and *PyEnchant* can help us with that

```python
import nltk
import enchant
from string import punctuation
from enchant.checker import SpellChecker

sw = nltk.corpus.stopwords.words('english')
checker=SpellChecker('en_US')

docset = ['...','...',...] # your corpus
```
Now, for every document in your corpus you can run the following code to define its vocabulary.
```python
checker.set_text(text)
errors = [err.word for err in checker]
vocab = [word.strip(punctuation) for word in nltk.wordpunct_tokenize(text) if word.strip(punctuation) not in sw+errors]
vocab = list(set(vocab))
```
Now that you have a vocabulary, which the union of all the vocabularies of each document, you can run the 
LDA analysis. You have to specify the number of topics you expect to find (K below)
```python
K=10
D = 100 #Number of documents in the docset
olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024, 0.7)
for doc in docset:
  gamma, bound = olda.update_lambda(doc)
  wordids, wordcts = onlineldavb.parse_doc_list(doc,olda._vocab)
  perwordbound = bound * len(docset) / (D*sum(map(sum,wordcts)))
np.savetxt('lambda.dat',olda._lambda)
```

Finally you can visualize the resulting topics as a Word Cloud:
```python
cloud = GenCloud(vocab,lamb)
for i in range(K):
  cloud.gen_image(i)
```
If you have done everything right you should see a figure just like this:

(https://github.com/NAMD/topicmodeling/blob/master/tests/topic_0.png?raw=true)

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
```
Now that you have a vocabulary, which the union of all the vocabularies of each document, you can run the 
LDA analysis.


topicmodeling
=============

Library containing tools for topic modeling and related NLP tasks.

It brings together implementations from various authors, slightly modified by me as well as a new visualization tools
to help inspect the results.

I have also added a fair ammount of tests, mainly to guide my refactoring of
the code. Tests are still sparse, but will grow as the rest of the codebase sees more usage and refactoring.

Quick tutorial
--------------

Online LDA
the sub-package onlineldavb is currently  the most used/tested.
Here is a quick example of its usage:
Assume you have a set of documents you want to extract the most representative topics from. 




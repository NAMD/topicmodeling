#coding:utf8

import numpy as np
from Topics.visualization.wordcloud import make_wordcloud

class GenCloud(object):
    def __init__(self, vocab, topics):
        """
        Generates topic cloud image for the LDA topic model
        :param vocab: vocabulary list used to estimate topics
        :param topics: variational parameter numpy matrix, in which each line is a topic, with the coefficients
            for each word in the vocabulary.
        """
        self.vocab = np.array(vocab)
        self.topics = topics

    def gen_image(self, topic, fname='topic', width=600, height=400):
        """
        Generates and shows the image for the topic specified
        :param width: width of the resulting image
        :param height: height of the resulting image
        :param fname: Name of the file with which to save the topic cloud images
        :param topic: Integer corresponding to the line of the topic matrix
        """
        make_wordcloud(words=self.vocab, counts=self.topics[topic, :], fname="{}_{}.png".format(fname, topic), width=width, height=height)



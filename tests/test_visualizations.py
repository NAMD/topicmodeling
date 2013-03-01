# coding: utf-8


import unittest
import os

import numpy as np

from Topics.visualization import topiccloud


class TestTopicClouds(unittest.TestCase):
    def setUp(self):
        with open('../Topics/onlineldavb/dictnostops.txt') as f:
            self.vocab = f.read().split()
        self.topics = np.loadtxt('../Topics/onlineldavb/lambda.dat')

    def test_gencloud_class_really_produces_images(self):
        GC = topiccloud.GenCloud(self.vocab, self.topics)
        GC.gen_image(0)
        assert os.path.exists('topic_0.png')
        # os.unlink('topic_0.png')

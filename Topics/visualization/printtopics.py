#!/usr/bin/python

# printtopics.py: Prints the words that are most prominent in a set of
# topics.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import numpy


def list_topics(vocab, testlambda):
    """

    :param vocab:
    :param testlambda:
    """
    for k in range(0, len(testlambda)):
        lambdak = list(testlambda[k, :])
        lambdak /= sum(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key=lambda x: x[0], reverse=True)
        print 'topic {0:d}:'.format(k)
        # feel free to change the "53" here to whatever fits your screen nicely.
        for i in xrange(0, 53):
            # try:
            print u'{0:>20s}  \t---\t  {1:.4f}'.format(vocab[temp[i][1]], temp[i][0])
            # except UnicodeEncodeError:
            #     print temp[i][1]
        print


def main():
    """
    Displays topics fit by onlineldavb.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    vocab = str.split(file(sys.argv[1]).read())
    testlambda = numpy.loadtxt(sys.argv[2])

    list_topics(vocab, testlambda)


if __name__ == '__main__':
    main()

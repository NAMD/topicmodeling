#! /usr/bin/python
# coding: utf-8

# (C) Copyright 2009, David M. Blei (blei@cs.princeton.edu)

# This file is part of TURBOTOPICS.

# TURBOTOPICS is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your
# option) any later version.

# TURBOTOPICS is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
# USA

from __future__ import division
import sys, random, copy, itertools
import regex
from math import pow, log, exp
import codecs
from multiprocessing import Pool
import nltk



_chi_sq_table = {0.1: 2.70554345,
                 0.01: 6.634897,
                 0.001: 10.82757,
                 0.0001: 15.13671,
                 0.00001: 19.51142,
                 0.000001: 23.92813,
                 0.0000001: 28.37399}

_stop_words = []
# -------------------------------------------------------------------------


class Counts:
    """
    a class for word counts.

    this packages up
    - marginal counts
    - marginal next word counts
    - bigram counts
    - a vocabulary machine with multi token terms like "new york"
    """

    def __init__(self):
        self.vocab = {}
        self.reset_counts()
        self.marg = {}
        self.next_marg = {}
        self.bigram = {}

    def update_counts(self, doc, root_filter=lambda x: True, next_filter=lambda x: True):
        """
        Updates the bigram and marginal counts with a line of text.
        takes two filter functions and does not count words or next
        words if they do not pass through the filter.  (e.g., the
        filter can be used to remove stop words.)

        :param doc:
        :param root_filter:
        :param next_filter:
        """
        words = word_list(doc, self.vocab)
        for pos in range(len(words)):
            w = words[pos]
            if not root_filter(w):
                continue
            self.marg[w] = self.marg.get(w, 0) + 1
            if pos == len(words) - 1:
                break
            w_next = words[pos + 1]
            if not next_filter(w_next):
                continue
            bigram_w = self.bigram.setdefault(w, {})
            bigram_w[w_next] = bigram_w.get(w_next, 0) + 1
            self.next_marg[w_next] = self.next_marg.get(w_next, 0) + 1

    def reset_counts(self):
        """reset the counts"""
        self.marg = {}
        self.next_marg = {}
        self.bigram = {}

    def sig_bigrams(self, word, sig_test, Min, recursive=True):
        """
        Computes significant bigrams from a word.
        :rtype : dict
        :param word: Word contained in the bigram
        :param sig_test:
        :param Min:
        :param recursive:
        requires a significance tester object with:
        - score : (next_marg, next_bigram) -> [words->reals]
        - null_score : (next_marg, next_bigram, pvalue) -> null_value
        """
        if word not in self.bigram:
            return {}
        marg = copy.deepcopy(self.next_marg)
        bigram_w = copy.deepcopy(self.bigram.get(word, {}))
        marg_w = sum(bigram_w.values())
        total = sum(marg.values())
        selected = {}
        out = sys.stdout.write
        scores = sig_test.score(marg_w, marg, bigram_w, total, Min)
        scores = sorted(scores.items(), key=lambda x: -x[1])
        for cand, max_score in [s for s in scores if s[1] > 0]:
            if bigram_w[cand] < Min:
                continue
            null_score = sig_test.null_score(marg_w, marg, total)
            out('%-20s: marg = [%6d, %6d]; bigram = %5d;' %
                ("%s %s" % (word, cand), marg_w, marg[cand], bigram_w[cand]))
            out('val = %3.2e; null = %3.2e' % (max_score, null_score))
            if max_score <= null_score:
                out(' rejected\n')
            else:
                new_word = '%s %s' % (word, cand)
                selected[new_word] = bigram_w[cand]
                out(' selected *\n')
            if recursive:
                marg_w -= bigram_w[cand]
                total -= bigram_w[cand]
                del(bigram_w[cand])

        return selected

# -------------------------------------------------------------------------

class LikelihoodRatio:
    def __init__(self, pvalue, use_perm, perm_hash=10):
        self.pvalue = pvalue
        self.perms = {}
        self.perm_hash = perm_hash
        if use_perm:
            self.null_score = self.null_score_perm
        else:
            self.null_score = self.null_score_chi_sq

    def reset(self):
        """reset the permutation cache"""

        self.perms = {}


    def score(self, count, unigram, bigram, total, min_count):
        """
        :param count:
        :param unigram:
        :param bigram:
        :param total:
        :param min_count:
        input:
        - count of root word (scalar)
        - unigram counts of next word (dictionary)
        - bigram counts of next word (dictionary)
        - total words (of next word) (scalar)

        for each bigram, compute 2 times the log likelihood ratio of
        modeling it as a bigram versus not modeling it as a bigram
        """

        def mylog(x):
            if x == 0: return -1000000
            return log(x)

        val = {}
        for v in bigram.keys():
            uni = unigram.get(v, 0)
            big = bigram.get(v, 0)
            if big < min_count: continue
            assert (uni >= big)

            log_pi_vu = mylog(big) - mylog(count)
            log_pi_vnu = mylog(uni - big) - mylog(total - big)
            log_pi_v_old = mylog(uni) - mylog(total)
            log_1mp_v = mylog(1 - exp(log_pi_vnu))
            log_1mp_vu = mylog(1 - exp(log_pi_vu))

            val[v] = 2 * (big * log_pi_vu + \
                          (uni - big) * log_pi_vnu - \
                          uni * log_pi_v_old + \
                          (count - big) * (log_1mp_vu - log_1mp_v))

            assert (val[v] == val[v])  # checks for nans

        return val

    def null_score_perm(self, count, marg, total):
        """returns the maximum maximum-score achieved by a permutation
        :param count:
        :param marg:
        :param total:
        """

        perm_key = int(count / self.perm_hash)
        if perm_key in self.perms:
            return self.perms[perm_key]
        max_score = 0
        nperm = int(1.0 / self.pvalue)
        table = sorted(marg.items(), key=lambda x: -x[1])
        for perm in xrange(nperm):
            perm_bigram = sample_no_replace(total, table, count)
            obs_score = self.score(count, marg, perm_bigram, total, 1)
            obs_score = max(obs_score.values())
            if obs_score > max_score or perm == 0:
                max_score = obs_score

        self.perms[perm_key] = max_score
        return max_score

    def null_score_chi_sq(self, count, marg, total):
        """returns the chi squared null score"""

        return _chi_sq_table[self.pvalue]


# -------------------------------------------------------------------------

class ChiSq:
    def __init__(self, pvalue):
        assert isinstance(pvalue, float)
        self.pvalue = pvalue

    def score(self, count, marg, bigram, total, min_count):
        """
        returns the chi_sq test scores
        :rtype : Dictionary
        :param count:
        :param marg:
        :param bigram: dictionary with frequencies of bigram
        :param total:
        :param min_count: minimum frequency for bigram
        """
        scores = {}
        for w2 in bigram.keys():
            if bigram[w2] < min_count:
                continue
            o_11 = bigram[w2]
            o_12 = marg[w2] - bigram[w2]
            o_21 = count - bigram[w2]
            o_22 = total - count - marg[w2]
            num = float(total * pow(o_11 * o_22 - o_12 * o_21, 2))
            den = float((o_11 + o_12) * (o_11 + o_21) * (o_12 + o_22) * (o_21 + o_22))
            scores[w2] = num / den

        return scores

    def null_score(self, count, marg, total):
        """returns the chi squared null score"""

        return _chi_sq_table[self.pvalue]


# !!! note: we should do the permtest thing a bit better...

class MultTest:
    """multinomial likelihood ratio significance test"""

    def __init__(self, pvalue, use_perm, perm_hash=10):
        self.pvalue = pvalue
        self.perms = {}
        self.perm_hash = perm_hash
        if use_perm:
            self.null_score = self.null_score_perm
        else:
            self.null_score = self.null_score_chi_sq

    def score(self, count, marg, bigram, total, min_count):
        """returns the GOF test scores
        :param count:
        :param marg:
        :param bigram:
        :param total:
        :param min_count:
        """
        scores = {}
        n_u = count
        n = total
        n_nu = n - n_u
        log_n_u = log(n_u)
        log_n = log(n)
        for v in bigram.keys():
            if (bigram[v] < min_count): continue
            n_v = marg[v]
            n_nv = n - n_v
            n_uv = bigram[v]
            n_nuv = n_v - n_uv
            n_unv = n_u - n_uv
            n_nunv = n - n_u - (n_v - n_uv)
            val = 0
            if (n_uv > 0):
                val += (n_uv) * (log(n_uv) - log_n_u - log(n_v) + log_n)
            if (n_nuv > 0):
                val += (n_nuv) * (log(n_nuv / n) - log(n_nu / n) - log(n_v / n))
            if (n_unv > 0):
                val += (n_unv) * (log(n_unv) - log_n_u - log(n_nv) + log_n)
            if (n_nunv > 0):
                val += (n_nunv) * (log(n_nunv / n) - log(n_nu / n) - log(n_nv / n))
            scores[v] = 2 * val

        return (scores)


    def null_score_perm(self, count, marg, total):

        "returns the maximum maximum-score achieved by a permutation"

        perm_key = int(count / self.perm_hash)
        if (perm_key in self.perms): return (self.perms[perm_key])
        max_score = 0
        nperm = int(1.0 / self.pvalue)
        table = sorted(marg.items(), key=lambda x: -x[1])
        for perm in xrange(nperm):
            perm_bigram = sample_no_replace(total, table, count)
            obs_score = self.score(count, marg, perm_bigram, total, 1)
            obs_score = max(obs_score.values())
            if obs_score > max_score or perm == 0:
                max_score = obs_score

        self.perms[perm_key] = max_score
        return (max_score)


    def null_score_chi_sq(self, count, marg, total):
        """
        returns the chi squared null score
        :param count:
        :param marg:
        :param total:
        """

        return (_chi_sq_table[self.pvalue])


###########################################################################
#
# input and output
#

def write_vocab(v, outfname, incl_stop=False):
    """writes a file of terms and counts
    :param v:
    :param outfname:
    :param incl_stop: Boolean. Wheter to include stopwords in the vocab file.
    """
    if not v:  # if dictionary is empty
        return
    with codecs.open(outfname, 'w', encoding='utf8') as f:
        [f.write('%-25s | %8.2f\n' % (i[0].strip(), i[1])) for i in sorted(v.items(), key=lambda x: -x[1])
         if incl_stop or i[0] not in _stop_words]


###########################################################################
#
# helper functions
#

def sample_no_replace(total, table, nitems):
    """
    sample without replacement from a list of items and counts
    :param total:
    :param table:
    :param nitems:
    """

    def nth_item_from_table(n):
        Sum = 0
        for i in table:
            Sum = Sum + i[1]
            if n < Sum:
                return i[0]
        print(n)
        assert (False)

    sample = random.sample(xrange(total), nitems)
    count = {}
    for n in sample:
        w = nth_item_from_table(n)
        count[w] = count.get(w, 0) + 1
    return count


def word_list(doc, vocab):
    """
    :param doc: document on a line
    :param vocab: vocabulary "machine", e.g.,
      {'new':{'york':{}},
       'long':{'island': {'city':{}, 'railroad':{}}}})

    output:
    - list of words in that document
      note: n-grams are matched from left to right and longest
    """
    doc = strip_text(doc)
    singles = doc.split(' ')
    words = []
    pos = 0
    while pos < len(singles):
        w = singles[pos]
        pos += 1
        word = w
        state = vocab.setdefault(w, {})
        while pos < len(singles) and state.has_key(singles[pos]):
            state = state[singles[pos]]
            word += ' ' + singles[pos]
            pos += 1
        words.append(word)

    return words


def strip_text(text):
    """
    strips out all non alphabetic characters from a string,
    lower cases it, and removes extra whitespace characters.
    :param text: Text to be stripped
    """

    text = text.lower()
    text = regex.sub(u"_", u" ", text)
    text = regex.sub(ur"[^\p{L}\p{N} ]+", u"", text)
    text = regex.sub(ur"\s+", u" ", text)
    text = text.strip()
    return text


def words_from_vocab_machine(mach):
    """
    Recursively generate all possible words from a vocabulary machine.
    :rtype : list
    :param mach: Dictionary
    """
    words = []
    for v_1, next_mach in mach.iteritems():
        words.append(v_1)
        words.extend(['%s %s' % (v_1, v_2)
                      for v_2 in words_from_vocab_machine(next_mach)])
    return words


def nested_sig_bigrams(iter_generator, update_fun, sig_test, Min):
    """
    finds nested significant bigrams.
    given a function to produce an iterator
    and a function to update the counts based on that iterator,
    :rtype : Counts object
    :param iter_generator: Generator function
    :param update_fun: 
    :param sig_test: 
    :param Min:
    """

    sys.stdout.write("computing initial counts\n")
    # po = Pool()
    counts = Counts()
    for doc in iter_generator():
        update_fun(counts, doc)
    terms = [item[0] for item in
             sorted(counts.marg.items(), key=lambda x: -x[1])
             if item[1] >= Min]
    while terms:
        new_vocab = {}
        sig_test.reset()
        sys.stdout.write("analyzing %d terms\n" % len(terms))

        # args = [(counts, v, sig_test, Min) for v in terms]
        # sig_bigrams = po.imap_unordered(counts.sig_bigrams, args)
        # new_vocab.update(sig_bigrams)
        for v in terms:
            sig_bigrams = counts.sig_bigrams(v, sig_test, Min)
            new_vocab.update(sig_bigrams)

        for selected in new_vocab.keys():
            sys.stdout.write("bigram : %s\n" % selected)
            update_vocab(selected, counts.vocab)

        # reset counts
        counts.reset_counts()
        for doc in iter_generator():
            update_fun(counts, doc)
        terms = [item[0] for item in
                 sorted(new_vocab.items(), key=lambda x: -x[1])
                 if item[1] >= Min]
    # po.close()
    # po.join()
    return counts


# -------------------------------------------------------------------------
def update_vocab(word, vocab):
    """updates a vocabulary transition machine with an n-gram"""
    words = word.split()
    mach = vocab
    i = 0
    for w in words:
        if w not in mach:
            mach.update({w: {}})
        mach = mach[w]
        i += 1


def stop_filter(w):
    return w not in _stop_words


def make_char_filter(n):
    return lambda w: len(w) >= 2


def digit_filter(w):
    if regex.search("[0-9]", w):
        return False
    else:
        return True

# !!! write a number filter and a short character filter

def testme():
    corpus_filename = "ap/ap.txt"
    count = Counts()
    with open(corpus_filename) as f:
        count.update_counts(f.read(), root_filter=stop_filter)
    return count

#~ lr = LikelihoodRatio(0.1,0)
#~ cs = ChiSq(0.1)
#~ count = Counts()
#~ count.sig_bigrams('new', lr.score, lr.null_score_chi_sq, 5)
#~ count.sig_bigrams('new', lr.score, lr.null_score_perm, 5)
#~ count.sig_bigrams('new', cs.score, cs.null, 5

#~ count = testme()
if __name__ == "__main__":
    import pprint
    import cProfile

    command = "testme()"
    cProfile.runctx(command, globals(), locals(), filename="OpenGLContext.profile")
    #pprint.pprint(count.vocab)


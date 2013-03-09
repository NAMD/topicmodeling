# coding: utf-8


import unittest
import Topics.visualization.turbotopics as tt
import regex
from glob import glob
import codecs


class TestTurboTopics(unittest.TestCase):
    def test_strip_text_handles_utf8(self):
        text = u"Meu Cão é $paraplégico#."
        stext = tt.strip_text(text)
        self.assertEqual(regex.sub(ur"[^\p{L}\p{N} ]+", u"", text), u"Meu Cão é paraplégico")
        self.assertEqual(stext, u"Meu Cão é paraplégico".lower())

    def test_nested_sig_bigram_returns_counts_object(self):
        fns = glob('data/corpus_pt_BR/*.txt')
        corpus = []
        for fn in fns:
            with codecs.open(fn, encoding='utf8') as f:
                corpus.append(f.read())
        def iter_gen():
            for doc in corpus:
                yield doc

        def update_fun(count, doc):
            count.update_counts(doc, root_filter=tt.stop_filter)

        ### compute significant n-grams
        lr = tt.LikelihoodRatio(pvalue=0.01, use_perm=False)
        cnts = tt.nested_sig_bigrams(iter_gen, update_fun, lr, 5)
        self.assertIsInstance(cnts, tt.Counts)

# coding: utf-8


import unittest
import Topics.visualization.turbotopics as tt
import regex


class TestTurboTopics(unittest.TestCase):
    def test_strip_text_handles_utf8(self):
        text = u"Meu Cão é $paraplégico#."
        stext = tt.strip_text(text)
        self.assertEqual(regex.sub(ur"[^\p{L}\p{N} ]+", u"", text), u"Meu Cão é paraplégico")
        self.assertEqual(stext, u"Meu Cão é paraplégico".lower())

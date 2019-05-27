from data_utils import *

class KeywordExtractor():
    def __init__(self, idf_dict = None):
        self.idf_dict = idf_dict

    @staticmethod
    def is_keyword_tag(tag):
        return tag.startswith('VB') or tag.startswith('NN') or tag.startswith('JJ')

    @staticmethod
    def cal_tag_score(tag):
        if tag.startswith('VB'):
            return 1.
        if tag.startswith('NN'):
            return 2.
        if tag.startswith('JJ'):
            return 0.5
        return 0.

    def idf_extract(self, string, con_kw = None):
        tokens = simp_tokenize(string)
        seq_len = len(tokens)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        candi = []
        result = []
        for i, (word, tag) in enumerate(tokens):
            score = self.cal_tag_score(tag)
            if not is_candiword(source[i]) or score == 0.:
                continue
            if con_kw is not None and source[i] in con_kw:
                continue
            score *= source.count(source[i])
            score *= 1 / seq_len
            score *= self.idf_dict[source[i]]
            candi.append((source[i], score))
            if score > 0.15:
                result.append(source[i])
        return list(set(result))


    def extract(self, string):
        tokens = simp_tokenize(string)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        kwpos_alters = []
        for i, (word, tag) in enumerate(tokens):
            if source[i] and self.is_keyword_tag(tag):
                kwpos_alters.append(i)
        kwpos, keywords = [], []
        for id in kwpos_alters:
            if is_candiword(source[id]):
                keywords.append(source[id])
        return list(set(keywords))
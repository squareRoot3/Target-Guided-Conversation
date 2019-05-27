import nltk
import os
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()


def tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def kw_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])


def simp_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower])


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)


def truecasing(tokens):
    ret = []
    is_start = True
    for word, tag in tokens:
        if word == 'i':
            ret.append('I')
        elif tag[0].isalpha():
            if is_start:
                ret.append(word[0].upper() + word[1:])
            else:
                ret.append(word)
            is_start = False
        else:
            if tag != ',':
                is_start = True
            ret.append(word)
    return ret


candi_keyword_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'convai2/candi_keyword.txt')
_candiwords = [x.strip() for x in open(candi_keyword_path).readlines()]


def is_candiword(a):
    if a in _candiwords:
        return True
    return False


from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')


def calculate_linsim(a, b):
    linsim = -1
    syna = wn.synsets(a)
    synb = wn.synsets(b)
    for sa in syna:
        for sb in synb:
            try:
                linsim = max(linsim, sa.lin_similarity(sb, brown_ic))
            except:
                pass
    return linsim


def is_reach_goal(context, goal):
    context = kw_tokenize(context)
    for wd in context:
        if is_candiword(wd):
            rela = calculate_linsim(wd, goal)
            if rela > 0.9:
                return True
    return False


def make_context(string):
    string = kw_tokenize(string)
    context = []
    for word in string:
        if is_candiword(word):
            context.append(word)
    return context


def utter_preprocess(string_list, max_length):
    source, minor_length = [], []
    string_list = string_list[-9:]
    major_length = len(string_list)
    if major_length == 1:
        context = make_context(string_list[-1])
    else:
        context = make_context(string_list[-2] + string_list[-1])
    context_len = len(context)
    while len(context) < 20:
        context.append('<PAD>')
    for string in string_list:
        string = simp_tokenize(string)
        if len(string) > max_length:
            string = string[:max_length]
        string = ['<BOS>'] + string + ['<EOS>']
        minor_length.append(len(string))
        while len(string) < max_length + 2:
            string.append('<PAD>')
        source.append(string)
    while len(source) < 9:
        source.append(['<PAD>'] * (max_length + 2))
        minor_length.append(0)
    return (source, minor_length, major_length, context, context_len)

import inflect
import itertools
import logging
import re

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from utils.logging.helpers import log_and_warn

def format_and_regex(expr, special_delimiters=[], permutations=False, is_synonyms=False):
    
    regex_all_permutations = []
    
    if not isinstance(expr, str):
        logging.info("format_and_regex: wrong input type!")
        raise Exception("format_and_regex: wrong input type!")

    delimiter="\s"

    phrases = expr.split(r"[;]+")
        
    join_expr = '({0}+|{0}+.*{0}+)'.format(delimiter)
    
    regex_words = []
    
    phrases = [phrase.strip() for phrase in re.split(";+", expr.strip())]
    phrases = list(filter(None, phrases))
        
    for phrase in phrases:
        
        comma_separated_words = list(set(re.findall(r'([A-Za-z0-9/-]+)(?:[\s]*,[\s]*)', phrase)+\
                                         re.findall(r'(?:[\s]*,[\s]*)([A-Za-z0-9/]+)', phrase)))
        
        if permutations:
            expr_split_permutations = itertools.permutations(process_word(phrase, delimiter, special_delimiters, is_synonyms=is_synonyms))
            for perm in expr_split_permutations:
                regex_all_permutations.append('((^|.*{0}){1}({0}.*|$))'.format(delimiter, join_expr.join(perm)))

            regex_words.append("|".join(regex_all_permutations))
        else:
            expr_split_permutations = process_word(phrase, delimiter, special_delimiters, is_synonyms=is_synonyms)
            regex_words.append('((^|.*{0}){1}({0}.*|$))'.format(delimiter, join_expr.join(expr_split_permutations)))
        
    return "|".join(regex_words)

def pluralize(word):
    
    engine = inflect.engine()
    return engine.plural(word)

def lemmatize(word):

    lm = WordNetLemmatizer()
    return lm.lemmatize(word)

def stem(word):

    ps = PorterStemmer()
    return ps.stem(word)

def synonyms(word):

    word_synonyms = [word]

    try:
        word_synonyms = wn.synset('{}.n.01'.format(word)).lemma_names()
    except Exception:
        try:
            word_synonyms = wn.synset('{}.a.01'.format(word)).lemma_names()
        except Exception:
            try:
                word_synonyms = wn.synset('{}.r.01'.format(word)).lemma_names()
            except Exception:
                log_and_warn("No synonyms found for word {}!".format(word))

    word_synonyms = [word for word in word_synonyms if "_" not in word]

    return word_synonyms

def process_word(expr, delimiter, special_delimiters=[], is_synonyms=False):
    
    word_regex = []

    words = re.split(delimiter, expr)
    
    for word in words:
        word_list = [word, pluralize(word), stem(word), lemmatize(word)]
        if is_synonyms:
            word_list.extend(synonyms(word))
        word_list = list(set([word.lower() for word in word_list]))
        word_list = list(filter(None, word_list))
        word_regex.append("({})".format("|".join("(?i:{})".format(w) for w in word_list)))
        
    return word_regex

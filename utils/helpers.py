import re

def format_and_regex(expr):
    
    if len(expr.split()) <= 1:
        return expr
        
    regex_words = ["(?i:{})".format(w) for w in expr.split()]
        
    return "(^.*" + ".*".join(regex_words) + ".*$)"
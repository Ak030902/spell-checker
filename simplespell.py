import re
from collections import Counter


def tokenize(text):
    """
    Extract alphanumeric tokens.
    """

    return re.findall(r'\w+', text.lower())


with open('combined.txt', 'r') as f:
    WORDS = Counter(tokenize(f.read()))

N = sum(WORDS.values())


def probability(word):
    """
    Probability  for ´word´.
    """
    return WORDS[word]/N


def known(words):
    """
    The subset of ´words´ contained in the WORDS counter dictionary.
    """

    return set(w for w in words if w in WORDS)


def candidates(word):
    """
    Generates all possible corrections for ´word´.
    """
    return list((known([word]) or known(edits1(word)) or known(edits2(word)) or [word]))


def correction(word):
    """
    Most probable spelling correction for ´word´.
    """
    aa=candidates(word)
    try:
        aa.remove('th')
        aa.remove('ar')
    except:
        pass
    new=[]
    for x in aa:
        if len(list(x))>2:
            new.append(x)
    if len(new)>0:
        return new
    else:
        return aa


def edits1(word):
    """
    All edits that are one edit away from `word`.
    """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
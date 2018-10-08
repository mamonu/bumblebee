
import string


def string_matching_knuth_morris_pratt(text="", pattern=""):
    """Returns positions where pattern is found in text
       Ref: https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
       based on  Knuth, D. E., Morris, Jr, J. H., & Pratt, V. R. (1977),
            'Fast pattern matching in strings'. SIAM journal on computing, 6(2), 323-350.


       for text m and pattern n
       perform exact pattern matching in O(m) time rather than in O(m*n) 
        
       Example: 
    
        text = 'ababbababa', pattern = 'aba'
        string_matching_knuth_morris_pratt(text, pattern) returns [0, 5, 7]

    
    Args:
        text (str): text to search inside
        pattern (str): pattern string to search for

    Returns:
        list: containing offsets (shifts) where pattern is found inside text

    """
    n = len(text)
    m = len(pattern)
    offsets = []
    pi = kmp_prefix_function(pattern)
    q = 0
    for i in range(n):
        while q > 0 and pattern[q] != text[i]:
            q = pi[q - 1]
        if pattern[q] == text[i]:
            q = q + 1
        if q == m:
            offsets.append(i - m + 1)
            q = pi[q - 1]

    return offsets


def kmp_prefix_function(p):
    ### https://www.coursera.org/lecture/algorithms-on-strings/computing-prefix-function-5lDsK

    m = len(p)
    pi = [0] * m
    k = 0
    for q in range(1, m):
        while k > 0 and p[k] != p[q]:  # pragma: no cover
            k = pi[k - 1]
        if p[k] == p[q]:
            k = k + 1
        pi[q] = k
    return pi


def firstoccurence(text="", pattern=""):  # pragma: no cover
    # this function is needed for feature engineering on ML text data sometimes
    occurence = string_matching_knuth_morris_pratt(text, pattern)
    if occurence == []:
        out = -5000
    else:
        out = occurence[0]
    return out


def lastoccurence(text="", pattern=""):  # pragma: no cover
    # this function is needed for feature engineering on ML text data sometimes
    occurence = string_matching_knuth_morris_pratt(text, pattern)
    if occurence == []:
        out = -5000
    else:
        out = occurence[-1]
    return out

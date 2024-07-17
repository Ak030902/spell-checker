import simplespell
from nltk import bigrams
import numpy as np, re, sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import polars as pd
import polars as pl
from fuzzyset import FuzzySet
from rapidfuzz import process, utils, fuzz
#from two_lists_similarity import Calculate_Similarity as cs
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import collections
import time
import joblib
#sound = pl.read_csv("sound.csv")
#sound.set_index('word')
#sound_word = pl.read_csv("sound_inverse.csv")
#sound_word.set_index('sound')
fs = FuzzySet()
def ngrams(string, n=3):
    string = (re.sub(r'[,-./]|\sBD',r'', string)).upper()
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(M, N))


def get_matches_df(sparse_matrix, A, B, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = min(top, sparsecols.size)
    else:
        nr_matches = sparsecols.size

    if nr_matches <= 0:
        return pd.DataFrame(columns=['left_side', 'right_side', 'similarity'])

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)

    for index in range(nr_matches):
        if sparserows[index] < len(A) and sparsecols[index] < len(B):
            left_side[index] = A[sparserows[index]]
            right_side[index] = B[sparsecols[index]]
            similarity[index] = sparse_matrix.data[index]

    matches_df = pd.DataFrame({
        'left_side': left_side,
        'right_side': right_side,
        'similarity': similarity
    })

    return matches_df

#df = pd.read_pickle('multi_roman.pkl')
df = pl.read_csv("multi_roman.csv")
#df.to_csv('multi_roman.csv')
#df['eng']= df['eng'].astype(str)

a=df['eng'].to_list()
a=list(set(a))
#a=df.to_array(column_names=eng,strings=True,array_type=python)
df_clean = {"name":a}
for x in a:
    fs.add(x)
#print(len(a))



vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, sublinear_tf=True)
#vectorizer=joblib.load("vectorizer.pickle", 'rb')

#joblib.dump(vectorizer, open("vectorizer.pickle", "wb"))
tf_idf_matrix_clean = vectorizer.fit_transform(df_clean['name'])
tf_idf_matrix_clean=joblib.load("tf_idf_matrix_clean.pickle", 'r')
#joblib.dump(tf_idf_matrix_clean, open("tf_idf_matrix_clean.pickle", "wb"))
def comprehension_flatten_lists(the_lists):
    return [val for _list in the_lists for val in _list]
def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key

#while True:
def main(text):
    #text=input('Enter: ')
    start_time = time.time()
    sound_toks=[]
    toks=text.split()

    for x in toks:
        sound_toks.append(simplespell.correction(x))
    #sound_texts=' '.join(sound_toks)
    sound_toks = sum(sound_toks,[])
    t1 = time.time()
    n_grams = bigrams(text)
    to_list=[ ' '.join(grams) for grams in n_grams]
    len_text=len(text.split())
    topn=800*len_text
    df_dirty = {"name":[text]}
    tf_idf_matrix_dirty = vectorizer.transform(df_dirty['name'])
    matches = awesome_cossim_top(tf_idf_matrix_dirty, tf_idf_matrix_clean.transpose(), topn, 0)
    matches_df = get_matches_df(matches, df_dirty['name'], df_clean['name'], top=topn)
    print(matches_df)
    a=matches_df['right_side'].to_list()
    a=list(set(a))
    output = process.extract(text, a, scorer=fuzz.ratio, processor=None, limit=int(len(a)*.1))
    print(output)
    '''
    output_toks=[]
    for x in output:
        a=x[0].split()
        output_toks.append(a[0])
        output_toks.append(a[1])
    output_toks=list(set(output_toks))

    final_reduced=[]
    sound=[]
    result=[]
    for m in output_toks:
        sound.append(pysoundex.soundex(m))
    inverse = dict(zip(output_toks, sound))
    dictionary  = dict(zip(sound, output_toks))
    final_reduced=[]
    for x in text.split():
        output1 = process.extract(x, output_toks, scorer=fuzz.ratio, processor=None, limit=10)
        tem=[]
        for z in output1:
            tem.append(z[0])
        tem.extend(sound_toks)
        fs = FuzzySet(tem)
        aa=fs.get(x)
        print(aa,'====',x)
        temp=[]
        for y in aa:
            temp.append(y[1])
        final_reduced.append(temp)
        #print('--------------------------------')
    print("--- %s seconds ---" % (time.time() - start_time))
    '''
    return output

while True:
    text=input('Enter: ')

    start_time = time.time()
    print(fs.get(text))
    print("--- %s seconds ---" % (time.time() - start_time))
    output = process.extract(text, a, scorer=fuzz.WRatio, processor=None, limit=500)
    print(output)
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(main(text))

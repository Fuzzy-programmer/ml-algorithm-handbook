"""NLP basics (executed): tokenization, normalization, Zipf/Heaps, BoW/TF-IDF, classification ablations, LSA, topic models, n-gram LMs."""
import re, math, collections, numpy as np, warnings; warnings.filterwarnings("ignore")
import nltk
from nltk.corpus import movie_reviews, brown, stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("== tokenization contrasts ==")
s = "Dr. Smith didn't pay $3.50 for U.S. e-mail; it's well-known... #NLP @user https://x.co"
print("whitespace:", s.split())
print("regex \\w+ :", re.findall(r"\w+", s))
print("nltk word_tokenize:", nltk.word_tokenize(s))
print("sentences:", nltk.sent_tokenize("Dr. Smith arrived at 5 p.m. He left. Mr. Lee stayed."))
ps, wl = PorterStemmer(), WordNetLemmatizer()
for w in ["running", "studies", "better", "happily", "universities", "was"]:
    print(f"{w:12s} stem={ps.stem(w):10s} lemma(v)={wl.lemmatize(w, 'v'):8s} lemma(n)={wl.lemmatize(w):12s}")

print("\n== Zipf and Heaps on Brown corpus ==")
toks = [w.lower() for w in brown.words() if w.isalpha()]; cnt = collections.Counter(toks); print("tokens", len(toks), "types", len(cnt))
for r in [1, 10, 100, 1000, 10000]:
    w, f = cnt.most_common(r)[-1]; print(f"rank {r:5d} word {w:12s} freq {f:6d} rank*freq {r*f}")
hap = sum(1 for f in cnt.values() if f == 1); print(f"hapax legomena: {hap} ({100*hap/len(cnt):.1f}% of types)")
for n in [10000, 100000, 500000]: print(f"first {n} tokens -> {len(set(toks[:n]))} types")

print("\n== TF-IDF worked example ==")
docs = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs"]
cv = CountVectorizer(); C = cv.fit_transform(docs); print(cv.get_feature_names_out().tolist()); print(C.toarray())
N = 3; df = (C.toarray() > 0).sum(0); idf_sk = np.log((1 + N) / (1 + df)) + 1
tf = TfidfVectorizer(norm=None).fit(docs); print("idf sklearn:", dict(zip(cv.get_feature_names_out(), tf.idf_.round(3))))
print("manual: idf('cat') = ln((1+3)/(1+1))+1 =", round(math.log(4 / 2) + 1, 3), "| idf('the') = ln(4/3)+1 =", round(math.log(4 / 3) + 1, 3))

print("\n== sentiment classification ablations on movie_reviews (2000 docs, 5-fold CV acc) ==")
texts = [movie_reviews.raw(f) for f in movie_reviews.fileids()]; y = np.array([1 if f.startswith("pos") else 0 for f in movie_reviews.fileids()])
cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
def acc(vec, clf): return cross_val_score(make_pipeline(vec, clf), texts, y, cv=cv5, n_jobs=1).mean()
stem_an = lambda d: [ps.stem(t) for t in re.findall(r"[a-z']+", d.lower())]
rows = [("majority", CountVectorizer(), None),
 ("BoW unigram + NB", CountVectorizer(), MultinomialNB()),
 ("BoW binary + NB", CountVectorizer(binary=True), MultinomialNB()),
 ("BoW unigram + LR", CountVectorizer(), LogisticRegression(max_iter=3000)),
 ("TF-IDF unigram + LR", TfidfVectorizer(sublinear_tf=True), LogisticRegression(max_iter=3000)),
 ("TF-IDF uni+bigram + LR", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=3000)),
 ("TF-IDF uni+bigram + LinearSVC", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), LinearSVC()),
 ("TF-IDF unigram, stopwords removed", TfidfVectorizer(sublinear_tf=True, stop_words="english"), LogisticRegression(max_iter=3000)),
 ("TF-IDF unigram, Porter stemming", TfidfVectorizer(sublinear_tf=True, analyzer=stem_an), LogisticRegression(max_iter=3000)),
 ("Hashing 2^16 (no idf) + LR, C=1", HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False), LogisticRegression(max_iter=3000)),
 ("Hashing 2^16 + tf-idf weighting + LR", make_pipeline(HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False, norm=None), TfidfTransformer(sublinear_tf=True)), LogisticRegression(max_iter=3000))]
for n, v, c in rows:
    if c is None: print(f"{n:34s} {max(y.mean(), 1 - y.mean()):.3f}"); continue
    print(f"{n:34s} {acc(v, c):.3f}")

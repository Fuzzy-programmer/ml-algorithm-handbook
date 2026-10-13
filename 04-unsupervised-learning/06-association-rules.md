# Chapter 4.6 — Association Rules: Apriori, FP-Growth, Eclat

[← Roadmap](../README.md) · Prev [4.5](05-anomaly-detection.md) · Next [4.7 Feature selection](07-feature-selection.md) · Priority ⚪ · Code: [`unsupervised_family.py`](../code/unsupervised_family.py)

## 1. Problem
Find items that frequently occur together in transactions ("market basket"): `{diapers} → {beer}`. Used for cross-sell, store layout, web-usage patterns, bundle design, log-pattern mining.

## 2. History
Association rules: Agrawal, Imielinski & Swami (1993). **Apriori**: Agrawal & Srikant (1994) — the downward-closure pruning idea. **Eclat**: Zaki (2000), vertical tid-list intersections. **FP-Growth**: Han, Pei & Yin (2000) — compresses the database into a prefix tree and avoids candidate generation. Successors: sequential pattern mining (GSP, PrefixSpan), and for recommendation, collaborative filtering/matrix factorization (Ch 6.2), which scale better and personalize.

## 3. Definitions
For itemsets A, B over N transactions:
`support(A) = #(A)/N`; `confidence(A→B) = support(A∪B)/support(A)`; `lift(A→B) = support(A∪B)/(support(A)·support(B))` (>1 positive association); also leverage, conviction.
**Apriori principle (downward closure):** every subset of a frequent itemset is frequent ⇒ if {A} is infrequent, no superset can be frequent ⇒ prune the search space.

## 4. Worked example
5 baskets: {bread, milk}; {bread, diaper, beer, eggs}; {milk, diaper, beer, cola}; {bread, milk, diaper, beer}; {bread, milk, diaper, cola}.
support(diaper) = 4/5 = 0.8; support(beer) = 0.6; support(diaper, beer) = 3/5 = 0.6.
Rule **diaper → beer**: confidence = 0.6/0.8 = **0.75**, lift = 0.6/(0.8·0.6) = **1.25**. With minimum support 0.4 the frequent pairs include {beer,diaper}(0.6), {bread,diaper}(0.6), {bread,milk}(0.6), {diaper,milk}(0.6) (verified by the code). Five baskets are far too few to trust — real mining needs thousands.

## 5. Algorithms and cost
| Algorithm | How | Strength | Weakness |
|---|---|---|---|
| Apriori | Level-wise candidate generation + pruning, multiple DB scans | Simple, educational | Many scans, candidate explosion |
| FP-Growth | FP-tree, recursive conditional mining | Usually fastest, two scans | Memory for tree |
| Eclat | Vertical format, intersect transaction-id sets | Fast on dense small data | Memory of tid-lists |
Worst case exponential in the number of items; cost controlled mainly by min support.

## 6. When to use / not
Use: exploratory basket analysis, rule-based recommendations, interpretable patterns, discovering co-occurrence in logs/alarms. Avoid: personalized recommendation at scale (use CF/MF), causal claims ("beer causes diapers"), very low support thresholds (explosion), sparse long-tail catalogs (most items below support), noisy time-ordered behavior (use sequence models).

## 7. Hyperparameters & selection
`min_support` (main control; start high), `min_confidence`, `min_lift`, `max_len`. Filter rules by lift and by business actionability; beware **spurious** rules from multiple testing and popular-item bias (high-confidence rules into a very popular consequent have lift ≈ 1).

## 8. Implementation
```python
# pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
te = TransactionEncoder(); df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
freq = fpgrowth(df, min_support=0.4, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.0)
```
From scratch: count single items, keep those ≥ min support, join frequent k-itemsets into (k+1)-candidates, prune candidates with an infrequent subset, repeat (see the toy Apriori in `unsupervised_family.py`).

## 9. Evaluation
Rules are descriptive, so evaluate by **business impact**: A/B test cross-sell placements; hold-out time period to check rule stability (support/confidence persistence); precision of top-k rule-based recommendations vs a popularity baseline.

## Interview Q&A
- *Support/confidence/lift?* Frequency, conditional probability of consequent, and strength relative to independence.
- *Why Apriori prunes?* Downward closure: a superset cannot be more frequent than any subset.
- *Why lift over confidence?* Confidence is inflated when the consequent is simply popular.
- *Apriori vs FP-Growth?* Candidate generation with repeated scans vs tree compression with two scans.

## Exercises
Compute all rules from the toy baskets with lift > 1; implement Eclat with Python sets; compare runtime of Apriori vs FP-Growth as min-support drops.

## Further reading
Agrawal & Srikant (1994); Han, Pei & Yin (2000); Zaki (2000); Aggarwal, *Data Mining: The Textbook* (frequent patterns chapters); `mlxtend` docs.

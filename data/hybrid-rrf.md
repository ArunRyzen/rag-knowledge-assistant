# Hybrid retrieval and Reciprocal Rank Fusion

Hybrid retrieval runs two searches for every query — dense (embedding similarity) and sparse
(BM25 keyword matching) — and merges their results into a single ranking. The two methods fail
in different places: dense retrieval misses exact identifiers but catches paraphrases; sparse
retrieval nails exact terms but is blind to synonyms. Because their error patterns are
complementary, the combination consistently outperforms either method alone on mixed real-world
query loads, and hybrid has become the default architecture in production RAG.

## The score-fusion problem

Merging the two result lists is less obvious than it sounds, because the scores are not
comparable. Cosine similarity lives in a bounded range around 0 to 1, while BM25 scores are
unbounded and depend on corpus statistics — a BM25 score of 12 means nothing on the cosine
scale. Adding or averaging raw scores therefore requires fragile normalization with weights
that need constant re-tuning as the corpus changes.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) sidesteps score incompatibility by ignoring scores entirely and
using only each item's rank position. Every list votes for its items: an item at rank r in a
list receives 1/(k + r) points, where k is a smoothing constant conventionally set to 60. Sum
the points across lists and sort. An item ranked highly by BOTH retrievers collects two large
contributions and rises to the top; an item that only one retriever liked still survives, just
lower down.

The constant k controls how steep the drop-off is. With small k, rank 1 is worth vastly more
than rank 10 and the top of each list dominates. With large k, the difference between rank 1
and rank 10 flattens, treating each list more like a set of endorsements. The default of 60
comes from the original RRF paper and works remarkably well untouched — one of RRF's selling
points is that it has essentially one parameter and rarely needs tuning.

## Why rank-based fusion is robust

Ranks are scale-free: they are unaffected by whether a retriever's scores are between 0 and 1
or between 0 and 100, so RRF needs no normalization, no learned weights, and no assumptions
about score distributions. New retrievers can be added to the fusion — a third list from a
second embedding model, or from a query rewrite — without changing anything. The trade-off is
information loss: rank 1 with score 0.99 and rank 2 with score 0.98 are treated as far apart as
rank 1 and rank 2 with a huge score gap. In practice the robustness is worth far more than the
lost precision.

## Retrieve broadly, then cut

A practical detail: fusion works best when each retriever over-fetches. Pulling the top 20 from
each side before fusing, then cutting the fused list to the final 5, gives the fusion enough
raw material to surface items that one list ranked modestly but both lists agreed on.

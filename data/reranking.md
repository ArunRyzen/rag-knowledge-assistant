# Reranking with cross-encoders

Reranking is a second, more expensive scoring stage applied to the candidates that first-stage
retrieval returns. The pattern is: retrieve 20–50 candidates cheaply with dense and sparse
search, then re-score just those few with a slower, more accurate model and keep the best 5.
This two-stage funnel — recall first, precision second — is the same architecture web search
engines have used for decades.

## Bi-encoders versus cross-encoders

First-stage dense retrieval uses a bi-encoder: the query and each document are embedded
*separately*, and relevance is approximated by the similarity of the two independent vectors.
This is what makes it fast — document vectors are precomputed offline, and query time is one
embedding plus a nearest-neighbour lookup. But the model never sees the query and document
together, so it cannot judge fine-grained interactions between them.

A cross-encoder instead feeds the query and a candidate passage into the model *as one input*
and outputs a single relevance score. Every token of the query can attend to every token of the
passage, which captures negation, word order, and subtle mismatches that independent embeddings
blur. Accuracy is substantially higher — and cost is too: one full model inference per
query-passage pair, none of it precomputable, which is exactly why cross-encoders are reserved
for the shortlist rather than run over the whole corpus.

## What reranking buys

Reranking mostly improves the ORDER of results rather than finding new ones — it cannot recover
a document the first stage missed entirely. Its effect shows up in rank-sensitive metrics: MRR
and nDCG rise as the truly relevant passage moves from rank 4 to rank 1, while recall@k for
large k barely moves. That ordering matters because the top-ranked passages are what fills the
generator's limited context window, and language models pay more attention to context presented
first. A well-ranked top 5 beats a poorly-ordered top 20.

## Practical choices

Common open-weight cross-encoders include the MS MARCO MiniLM family from the
sentence-transformers library — small enough to run on CPU for moderate loads. Managed
alternatives such as Cohere Rerank or Voyage rerank models offer higher quality behind an API
call. The knobs that matter are how many candidates to rerank (more candidates, more latency,
diminishing returns past 50) and latency budget: a cross-encoder pass adds tens to hundreds of
milliseconds, which is fine for chat but may not be for autocomplete. When answer quality
depends on the single best passage being on top — long documents, subtle questions — reranking
is usually the cheapest big win available.

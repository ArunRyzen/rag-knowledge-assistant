# RAG concepts — one-page notes per topic

Everything you need to *explain* RAG, topic by topic. Read the section that matches the
code you're working with; the tasks in `tasks/README.md` tell you when.

## Embeddings and dense vector search

An embedding is a learned mapping from text to a fixed-length vector of numbers, produced by a
neural network trained so that texts with similar meaning land close together in vector space.
"The cat sat on the mat" and "a feline rested on the rug" share almost no words, yet a good
embedding model places their vectors near each other. This is the property that makes semantic
search possible: instead of matching words, we compare positions in a geometric space.

### How similarity is measured

The standard similarity measure is cosine similarity: the cosine of the angle between two
vectors. It ranges from -1 (opposite) through 0 (unrelated) to 1 (identical direction). When
vectors are L2-normalized — scaled to length one — cosine similarity reduces to a simple dot
product, which is fast to compute and easy to index. Most vector databases, including Pinecone,
default to cosine similarity for text embeddings.

### Dense retrieval

Dense retrieval embeds every document chunk ahead of time and stores the vectors in an index.
At query time the question is embedded with the same model, and the index returns the nearest
neighbours — the chunks whose vectors are most similar to the query vector. The word "dense"
distinguishes these vectors, where every dimension holds a meaningful value, from "sparse"
representations like bag-of-words, where almost every dimension is zero.

The strength of dense retrieval is recall on paraphrases: it finds relevant passages even when
the query uses entirely different words than the document. Its weakness is exact terms. Product
codes, function names, rare proper nouns, and version numbers are often poorly represented in
embedding space, and a lexical method like BM25 can beat dense retrieval on those queries. This
is the core motivation for hybrid retrieval.

### Practical decisions

Three choices matter most in practice. First, the embedding model: it determines quality and
cost, and query vectors must come from the same model as document vectors — mixing models
produces garbage similarities. Second, the dimension: Gemini's gemini-embedding-001 supports
768, 1536, or 3072 dimensions; higher dimensions capture more nuance but cost more storage and
compute, and 768 is a strong default. Third, normalization: normalize both document and query
vectors once at embedding time, so every comparison downstream is a plain dot product.

One operational trap: if you change the embedding model or dimension, every stored vector must
be re-embedded and re-indexed. The index dimension is fixed at creation time in most vector
databases, so a dimension change means creating a new index. Plan for re-indexing as a routine
operation, not an emergency.

---

## Chunking strategies

Chunking is the process of splitting documents into smaller passages before embedding and
indexing them. It is often called the highest-leverage decision in a RAG system: no retrieval
algorithm can rescue chunks that are cut badly, because the chunk is the unit of everything
downstream — it is what gets embedded, what gets retrieved, and what the language model reads.

### Why chunk size is a trade-off

Chunks that are too large hurt precision. A 3,000-character chunk about five topics embeds into
a vector that represents none of them well, and when retrieved it drags four irrelevant topics
into the model's context window. Chunks that are too small hurt comprehension: a single sentence
ripped from its section may be meaningless on its own — "It increased by 40%" retrieves nothing
useful without knowing what "it" was. A common starting point is 500 to 1,000 characters with
overlap, tuned by measuring retrieval quality, never by intuition.

### Overlap

Overlap repeats the tail of each chunk at the start of the next one, typically 10 to 20 percent
of the chunk size. Its purpose is to protect facts that straddle a boundary: if a key sentence
is cut in half at character 800, the overlap ensures at least one chunk contains the whole
sentence. The cost is a little extra storage and a chance of near-duplicate retrieval results,
which is why overlap stays small relative to chunk size.

### Structure-aware and recursive splitting

Naive splitting cuts every N characters regardless of content, slicing sentences mid-word. A
recursive, structure-aware splitter instead tries boundaries in order of preference: split on
paragraph breaks first; if a piece is still too large, split it on line breaks, then on sentence
boundaries, then on spaces as a last resort. This respects the document's natural structure —
paragraphs tend to be self-contained thoughts, so chunks aligned to them are more coherent.

More advanced strategies exist. Markdown-aware chunking keeps headings attached to their
sections and never splits a code block. Semantic chunking embeds each sentence and starts a new
chunk when similarity between consecutive sentences drops, finding topic boundaries
automatically at the cost of many embedding calls. Parent-document retrieval indexes small
chunks for precise matching but returns their larger parent section for fuller context.

### Measuring instead of guessing

Because chunking happens before everything else, its effects show up everywhere and are hard to
attribute. The reliable approach is to hold retrieval constant and vary only chunk size, then
compare recall@k and MRR on a labelled evaluation set. Halving or doubling the chunk size will
often move retrieval metrics more than switching retrieval algorithms — which is exactly why it
deserves to be measured first.

---

## BM25 and lexical search

BM25 (Best Matching 25, from the Okapi retrieval system) is the classic lexical ranking
function used by search engines for decades, and it remains the standard sparse complement to
dense retrieval in modern RAG systems. It scores a document for a query by combining three
signals: how often each query term appears in the document, how rare each term is across the
whole collection, and how long the document is.

### The three ingredients

Term frequency (TF) says a document mentioning "pinecone" five times is more relevant to a
pinecone query than one mentioning it once — but with diminishing returns. BM25 saturates term
frequency with the k1 parameter (typically 1.5): the fifth occurrence adds far less score than
the first, so keyword stuffing cannot dominate.

Inverse document frequency (IDF) weighs terms by rarity. A term appearing in nearly every
document, like "the", carries almost no information, while a term appearing in three documents
out of ten thousand is a powerful discriminator. IDF multiplies each term's contribution by a
logarithmic factor of its rarity, which is why rare product codes and error strings are exactly
where BM25 shines.

Length normalization, controlled by the b parameter (typically 0.75), gently penalizes long
documents, which would otherwise accumulate score simply by containing more words. With b=0 no
penalty is applied; with b=1 scores are fully normalized by document length.

### Where BM25 beats embeddings

Dense embeddings compress meaning into a fixed vector and can blur exact identifiers. Queries
like "error TS2345", "model gemini-embedding-001", or a person's surname are matched precisely
by BM25 because the literal token either appears in a document or it does not. Embeddings also
struggle with vocabulary they rarely saw in training; BM25 has no vocabulary — any token in the
corpus is searchable. The inverse also holds: BM25 knows nothing about synonyms or paraphrase,
so "automobile" never matches "car". Neither method dominates, which motivates running both.

### Sparse retrieval in practice

BM25 needs only tokenization and counting, no model or GPU, and an inverted index makes it fast
at scale: for each term, store the list of documents containing it, so only documents sharing at
least one query term are ever scored. In production stacks, lexical search is typically served
by Elasticsearch or OpenSearch, or by Postgres full-text search; small corpora can simply score
every document in memory. The scores BM25 produces are unbounded and corpus-dependent, which is
why fusing them with cosine similarities requires a rank-based method rather than adding raw
numbers together.

---

## Hybrid retrieval and Reciprocal Rank Fusion

Hybrid retrieval runs two searches for every query — dense (embedding similarity) and sparse
(BM25 keyword matching) — and merges their results into a single ranking. The two methods fail
in different places: dense retrieval misses exact identifiers but catches paraphrases; sparse
retrieval nails exact terms but is blind to synonyms. Because their error patterns are
complementary, the combination consistently outperforms either method alone on mixed real-world
query loads, and hybrid has become the default architecture in production RAG.

### The score-fusion problem

Merging the two result lists is less obvious than it sounds, because the scores are not
comparable. Cosine similarity lives in a bounded range around 0 to 1, while BM25 scores are
unbounded and depend on corpus statistics — a BM25 score of 12 means nothing on the cosine
scale. Adding or averaging raw scores therefore requires fragile normalization with weights
that need constant re-tuning as the corpus changes.

### Reciprocal Rank Fusion

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

### Why rank-based fusion is robust

Ranks are scale-free: they are unaffected by whether a retriever's scores are between 0 and 1
or between 0 and 100, so RRF needs no normalization, no learned weights, and no assumptions
about score distributions. New retrievers can be added to the fusion — a third list from a
second embedding model, or from a query rewrite — without changing anything. The trade-off is
information loss: rank 1 with score 0.99 and rank 2 with score 0.98 are treated as far apart as
rank 1 and rank 2 with a huge score gap. In practice the robustness is worth far more than the
lost precision.

### Retrieve broadly, then cut

A practical detail: fusion works best when each retriever over-fetches. Pulling the top 20 from
each side before fusing, then cutting the fused list to the final 5, gives the fusion enough
raw material to surface items that one list ranked modestly but both lists agreed on.

---

## Reranking with cross-encoders

Reranking is a second, more expensive scoring stage applied to the candidates that first-stage
retrieval returns. The pattern is: retrieve 20–50 candidates cheaply with dense and sparse
search, then re-score just those few with a slower, more accurate model and keep the best 5.
This two-stage funnel — recall first, precision second — is the same architecture web search
engines have used for decades.

### Bi-encoders versus cross-encoders

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

### What reranking buys

Reranking mostly improves the ORDER of results rather than finding new ones — it cannot recover
a document the first stage missed entirely. Its effect shows up in rank-sensitive metrics: MRR
and nDCG rise as the truly relevant passage moves from rank 4 to rank 1, while recall@k for
large k barely moves. That ordering matters because the top-ranked passages are what fills the
generator's limited context window, and language models pay more attention to context presented
first. A well-ranked top 5 beats a poorly-ordered top 20.

### Practical choices

Common open-weight cross-encoders include the MS MARCO MiniLM family from the
sentence-transformers library — small enough to run on CPU for moderate loads. Managed
alternatives such as Cohere Rerank or Voyage rerank models offer higher quality behind an API
call. The knobs that matter are how many candidates to rerank (more candidates, more latency,
diminishing returns past 50) and latency budget: a cross-encoder pass adds tens to hundreds of
milliseconds, which is fine for chat but may not be for autocomplete. When answer quality
depends on the single best passage being on top — long documents, subtle questions — reranking
is usually the cheapest big win available.

---

## Vector databases and Pinecone

A vector database stores embeddings and answers one query shape extremely well: "here is a
vector — return the k stored vectors nearest to it, with their metadata." Purpose-built systems
(Pinecone, Weaviate, Qdrant, Milvus) compete with extensions to general databases (pgvector for
Postgres) and lightweight libraries (FAISS, which is an index in your process rather than a
database). They differ in operations, not in the core idea.

### Approximate nearest neighbour and HNSW

Comparing a query against every stored vector — exact, brute-force search — is linear in
collection size and becomes too slow around the million-vector mark. Vector databases therefore
use Approximate Nearest Neighbour (ANN) indexes, trading a tiny amount of recall for orders of
magnitude in speed. The dominant algorithm is HNSW (Hierarchical Navigable Small World), a
multi-layer graph structure: sparse upper layers allow long hops across the space, dense lower
layers refine to the true neighbourhood, and a search greedily descends the layers. Queries run
in roughly logarithmic time, at the cost of holding the graph in memory and slower inserts.

### Records, metadata, and namespaces

The stored unit is a record: an id, the vector itself, and a metadata payload. Storing the
chunk's text and provenance (source document, position) in metadata means search results carry
everything needed to build a prompt, with no second lookup in another database. Metadata also
enables filtered search — "nearest neighbours WHERE doc_type = 'policy'" — which ANN indexes
support natively and which is essential for multi-tenant isolation and access control.
Namespaces partition one index into independent sub-collections, commonly one per tenant or per
environment.

### Pinecone specifics

Pinecone is a fully managed, serverless vector database: you create an index with a fixed
dimension and similarity metric, and capacity scales automatically with usage. Two properties
shape how applications are built on it. First, the dimension and metric are immutable after
index creation — changing embedding models to a different dimension means creating a new index
and re-ingesting. Second, writes are eventually consistent: an upserted record becomes
searchable after a short delay, typically seconds, so ingest-then-immediately-query can miss
fresh records. Upserts are idempotent by record id, which makes re-ingestion safe: writing the
same chunk id twice overwrites rather than duplicates.

### Choosing a backend

The honest decision tree is short. A prototype or small corpus fits in memory — numpy cosine
over a few thousand vectors runs in microseconds and needs no infrastructure. If your
application already runs Postgres, pgvector keeps vectors next to relational data with real
transactions and joins, and an HNSW index carries it to tens of millions of vectors. A managed
service like Pinecone removes index tuning, sharding, and capacity planning entirely, at the
price of a network hop per query, a per-usage bill, and your data living in a third-party
service. The retrieval logic above the store is identical in all three cases — which is why a
well-designed pipeline hides the choice behind an interface.

---

## Evaluating RAG systems

Without evaluation, every RAG decision — chunk size, retrieval mode, whether reranking is worth
its latency — is an argument settled by opinion. With a labelled dataset and two or three
metrics, the same decisions become measurements. Retrieval evaluation is the foundation because
retrieval quality caps everything downstream: a generator cannot cite a passage that was never
retrieved.

### The golden set

A golden set is a list of labelled examples: a realistic question paired with the identifier of
the document (or chunk) that answers it. Labelling relevance at the *document* level rather
than the chunk level keeps the labels stable when chunking parameters change — the same golden
set can compare chunk sizes, which chunk-level labels cannot. Even 20 to 50 carefully chosen
questions expose most retrieval regressions, though small sets make individual metric moves
noisy: with five questions, one flip changes recall by 0.2, so trends matter more than single
runs. Grow the set with every bug: each question a user asked that retrieval fumbled becomes a
permanent regression test.

### Recall@k and MRR

Recall@k asks a binary question per query: did any relevant document appear in the top k
results? It is the ceiling metric — if the answer is not in the retrieved set, no downstream
cleverness can recover it. Choose k to match how many contexts the generator actually receives.

MRR, Mean Reciprocal Rank, measures how HIGH the first relevant result ranks: reciprocal rank
is 1.0 for first place, 0.5 for second, 0.33 for third, averaged over queries. Two systems with
identical recall@5 can have very different MRR, and the one that puts the right passage first
wins in practice, because the model reads the top of the context most attentively. Together the
pair answers two separate questions: recall@k — "did we find it?"; MRR — "did we put it on
top?". nDCG generalizes MRR to graded, multi-document relevance but needs richer labels.

### Evaluating the generated answer

Retrieval metrics say nothing about what the model wrote. Answer evaluation adds at least two
more measurements. Faithfulness (groundedness): is every claim in the answer supported by the
retrieved passages? Unfaithful answers are hallucinations even when retrieval was perfect.
Answer relevance: does the answer actually address the question rather than summarizing nearby
context? Both are commonly scored by an LLM-as-judge — a strong model given the question,
contexts, and answer with a scoring rubric — which correlates well with human judgment when the
rubric is specific, though it costs a model call per example and inherits judge biases.
Frameworks like RAGAS package these metrics; the principle matters more than the tooling.

### Evaluation as regression testing

The habit that separates production teams: metrics run on every change, not once at the end.
Chunk size changed from 800 to 400? Run the eval. Swapped embedding models? Run the eval. A
one-command comparison of dense versus sparse versus hybrid, before and after a change, is the
RAG equivalent of a test suite — it converts "the retrieval feels better" into a number that
either moved or did not.

---

## Grounded generation and citations

Generation is the last stage of RAG: the model receives the retrieved passages and the user's
question, and must synthesize an answer *from the passages* — not from whatever it memorized
during training. This constraint is what makes RAG trustworthy. An ungrounded model answers
every question fluently, including the ones it knows nothing about; a grounded one is allowed
to say "the answer is not in my sources."

### The grounding prompt

Grounding is enforced primarily through the system prompt, which needs three specific
instructions. Answer ONLY from the provided context: this severs the model from its training
data for factual claims. If the answer is not in the context, say you don't know: without an
explicit refusal path, models fill gaps with plausible inventions — the refusal instruction
gives honesty a lower-energy path than hallucination. Cite the passages you used: passages are
numbered in the prompt, and the model marks claims with [1], [2], making every claim traceable
to a source. Low temperature (0 or near it) suits this task: grounded QA wants determinism and
precision, not creative variety.

### Prompt construction

The retrieved chunks are formatted into a numbered context block — each entry carrying its
number, source document id, and text — followed by the question. Numbering serves the citation
scheme; including the source id lets the model mention where information came from. Order
matters: models attend most reliably to the beginning of the context (and suffer "lost in the
middle" degradation on very long contexts), so the retriever's ranking should be preserved —
best passage first, and a context budget that stays well inside the model's window.

### Citations and verification

Citations turn an answer from an assertion into an argument: a reader (or an automated checker)
can open passage [2] and verify the claim against it. A subtle implementation point is that the
citations attached to an answer should reflect the passages the model actually cited in its
text, not simply every passage that was retrieved — parsing the [n] markers out of the answer
is the honest version. Stronger systems go further: a verification pass checks each cited
sentence against its cited passage and flags unsupported claims before the user sees them.

### Failure modes

Grounded generation fails in predictable ways worth testing deliberately. Ignoring the leash:
the model answers from parametric memory when retrieval returned weak passages — caught by
asking questions whose answers are absent from the corpus and checking for refusal. Citation
laundering: correct-looking citations attached to claims the passage does not support.
Retrieval poisoning: instructions embedded inside a retrieved document ("ignore your previous
instructions...") that the model obeys as if they were the user's — a prompt-injection channel
unique to RAG, mitigated by treating retrieved text as data and never as instructions. Each of
these is invisible in casual testing and obvious the moment you probe for it.

---

## Query transformation techniques

Retrieval quality depends on the query as much as on the index. Users write short, ambiguous
questions; documents are written in longer, more formal language. Query transformation rewrites,
expands, or reshapes the question *before* retrieval — applied at query time, no re-indexing,
often the cheapest way to lift retrieval quality.

**Query rewriting** asks a model to restate the question in a cleaner, more retrievable form —
fixing typos, expanding abbreviations, and (crucially, in chat) resolving pronouns from history:
"what does she eat?" must become "What does Chittu eat?" before retrieval can work. This is
exactly what the condenser agent in `chat.py` does.

**Multi-query expansion** generates 3–5 differently-worded variants, retrieves candidates for
each, and merges the lists with Reciprocal Rank Fusion — the same fusion already used for
dense + sparse. Each variant probes the embedding space from a different angle. Cost: one
generation call plus one retrieval per variant.

**HyDE (Hypothetical Document Embeddings)** flips the matching problem: ask the model to write a
*hypothetical answer* (invented facts are fine), embed that, and search with ITS vector. Because
the fake answer is shaped like a real answer, document-to-document similarity often beats
question-to-document similarity. The retrieved passages are real; only the search probe was
synthetic.

**Choosing:** rewriting is near-free hygiene; multi-query is the reliable recall booster; HyDE
shines when questions and documents use very different vocabulary. All add latency before
retrieval begins — measure the gain on a golden set before paying it in production.

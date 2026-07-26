# Embeddings and dense vector search

An embedding is a learned mapping from text to a fixed-length vector of numbers, produced by a
neural network trained so that texts with similar meaning land close together in vector space.
"The cat sat on the mat" and "a feline rested on the rug" share almost no words, yet a good
embedding model places their vectors near each other. This is the property that makes semantic
search possible: instead of matching words, we compare positions in a geometric space.

## How similarity is measured

The standard similarity measure is cosine similarity: the cosine of the angle between two
vectors. It ranges from -1 (opposite) through 0 (unrelated) to 1 (identical direction). When
vectors are L2-normalized — scaled to length one — cosine similarity reduces to a simple dot
product, which is fast to compute and easy to index. Most vector databases, including Pinecone,
default to cosine similarity for text embeddings.

## Dense retrieval

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

## Practical decisions

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

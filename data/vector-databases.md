# Vector databases and Pinecone

A vector database stores embeddings and answers one query shape extremely well: "here is a
vector — return the k stored vectors nearest to it, with their metadata." Purpose-built systems
(Pinecone, Weaviate, Qdrant, Milvus) compete with extensions to general databases (pgvector for
Postgres) and lightweight libraries (FAISS, which is an index in your process rather than a
database). They differ in operations, not in the core idea.

## Approximate nearest neighbour and HNSW

Comparing a query against every stored vector — exact, brute-force search — is linear in
collection size and becomes too slow around the million-vector mark. Vector databases therefore
use Approximate Nearest Neighbour (ANN) indexes, trading a tiny amount of recall for orders of
magnitude in speed. The dominant algorithm is HNSW (Hierarchical Navigable Small World), a
multi-layer graph structure: sparse upper layers allow long hops across the space, dense lower
layers refine to the true neighbourhood, and a search greedily descends the layers. Queries run
in roughly logarithmic time, at the cost of holding the graph in memory and slower inserts.

## Records, metadata, and namespaces

The stored unit is a record: an id, the vector itself, and a metadata payload. Storing the
chunk's text and provenance (source document, position) in metadata means search results carry
everything needed to build a prompt, with no second lookup in another database. Metadata also
enables filtered search — "nearest neighbours WHERE doc_type = 'policy'" — which ANN indexes
support natively and which is essential for multi-tenant isolation and access control.
Namespaces partition one index into independent sub-collections, commonly one per tenant or per
environment.

## Pinecone specifics

Pinecone is a fully managed, serverless vector database: you create an index with a fixed
dimension and similarity metric, and capacity scales automatically with usage. Two properties
shape how applications are built on it. First, the dimension and metric are immutable after
index creation — changing embedding models to a different dimension means creating a new index
and re-ingesting. Second, writes are eventually consistent: an upserted record becomes
searchable after a short delay, typically seconds, so ingest-then-immediately-query can miss
fresh records. Upserts are idempotent by record id, which makes re-ingestion safe: writing the
same chunk id twice overwrites rather than duplicates.

## Choosing a backend

The honest decision tree is short. A prototype or small corpus fits in memory — numpy cosine
over a few thousand vectors runs in microseconds and needs no infrastructure. If your
application already runs Postgres, pgvector keeps vectors next to relational data with real
transactions and joins, and an HNSW index carries it to tens of millions of vectors. A managed
service like Pinecone removes index tuning, sharding, and capacity planning entirely, at the
price of a network hop per query, a per-usage bill, and your data living in a third-party
service. The retrieval logic above the store is identical in all three cases — which is why a
well-designed pipeline hides the choice behind an interface.

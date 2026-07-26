# Chunking strategies

Chunking is the process of splitting documents into smaller passages before embedding and
indexing them. It is often called the highest-leverage decision in a RAG system: no retrieval
algorithm can rescue chunks that are cut badly, because the chunk is the unit of everything
downstream — it is what gets embedded, what gets retrieved, and what the language model reads.

## Why chunk size is a trade-off

Chunks that are too large hurt precision. A 3,000-character chunk about five topics embeds into
a vector that represents none of them well, and when retrieved it drags four irrelevant topics
into the model's context window. Chunks that are too small hurt comprehension: a single sentence
ripped from its section may be meaningless on its own — "It increased by 40%" retrieves nothing
useful without knowing what "it" was. A common starting point is 500 to 1,000 characters with
overlap, tuned by measuring retrieval quality, never by intuition.

## Overlap

Overlap repeats the tail of each chunk at the start of the next one, typically 10 to 20 percent
of the chunk size. Its purpose is to protect facts that straddle a boundary: if a key sentence
is cut in half at character 800, the overlap ensures at least one chunk contains the whole
sentence. The cost is a little extra storage and a chance of near-duplicate retrieval results,
which is why overlap stays small relative to chunk size.

## Structure-aware and recursive splitting

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

## Measuring instead of guessing

Because chunking happens before everything else, its effects show up everywhere and are hard to
attribute. The reliable approach is to hold retrieval constant and vary only chunk size, then
compare recall@k and MRR on a labelled evaluation set. Halving or doubling the chunk size will
often move retrieval metrics more than switching retrieval algorithms — which is exactly why it
deserves to be measured first.

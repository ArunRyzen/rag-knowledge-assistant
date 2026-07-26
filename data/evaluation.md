# Evaluating RAG systems

Without evaluation, every RAG decision — chunk size, retrieval mode, whether reranking is worth
its latency — is an argument settled by opinion. With a labelled dataset and two or three
metrics, the same decisions become measurements. Retrieval evaluation is the foundation because
retrieval quality caps everything downstream: a generator cannot cite a passage that was never
retrieved.

## The golden set

A golden set is a list of labelled examples: a realistic question paired with the identifier of
the document (or chunk) that answers it. Labelling relevance at the *document* level rather
than the chunk level keeps the labels stable when chunking parameters change — the same golden
set can compare chunk sizes, which chunk-level labels cannot. Even 20 to 50 carefully chosen
questions expose most retrieval regressions, though small sets make individual metric moves
noisy: with five questions, one flip changes recall by 0.2, so trends matter more than single
runs. Grow the set with every bug: each question a user asked that retrieval fumbled becomes a
permanent regression test.

## Recall@k and MRR

Recall@k asks a binary question per query: did any relevant document appear in the top k
results? It is the ceiling metric — if the answer is not in the retrieved set, no downstream
cleverness can recover it. Choose k to match how many contexts the generator actually receives.

MRR, Mean Reciprocal Rank, measures how HIGH the first relevant result ranks: reciprocal rank
is 1.0 for first place, 0.5 for second, 0.33 for third, averaged over queries. Two systems with
identical recall@5 can have very different MRR, and the one that puts the right passage first
wins in practice, because the model reads the top of the context most attentively. Together the
pair answers two separate questions: recall@k — "did we find it?"; MRR — "did we put it on
top?". nDCG generalizes MRR to graded, multi-document relevance but needs richer labels.

## Evaluating the generated answer

Retrieval metrics say nothing about what the model wrote. Answer evaluation adds at least two
more measurements. Faithfulness (groundedness): is every claim in the answer supported by the
retrieved passages? Unfaithful answers are hallucinations even when retrieval was perfect.
Answer relevance: does the answer actually address the question rather than summarizing nearby
context? Both are commonly scored by an LLM-as-judge — a strong model given the question,
contexts, and answer with a scoring rubric — which correlates well with human judgment when the
rubric is specific, though it costs a model call per example and inherits judge biases.
Frameworks like RAGAS package these metrics; the principle matters more than the tooling.

## Evaluation as regression testing

The habit that separates production teams: metrics run on every change, not once at the end.
Chunk size changed from 800 to 400? Run the eval. Swapped embedding models? Run the eval. A
one-command comparison of dense versus sparse versus hybrid, before and after a change, is the
RAG equivalent of a test suite — it converts "the retrieval feels better" into a number that
either moved or did not.

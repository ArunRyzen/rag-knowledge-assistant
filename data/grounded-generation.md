# Grounded generation and citations

Generation is the last stage of RAG: the model receives the retrieved passages and the user's
question, and must synthesize an answer *from the passages* — not from whatever it memorized
during training. This constraint is what makes RAG trustworthy. An ungrounded model answers
every question fluently, including the ones it knows nothing about; a grounded one is allowed
to say "the answer is not in my sources."

## The grounding prompt

Grounding is enforced primarily through the system prompt, which needs three specific
instructions. Answer ONLY from the provided context: this severs the model from its training
data for factual claims. If the answer is not in the context, say you don't know: without an
explicit refusal path, models fill gaps with plausible inventions — the refusal instruction
gives honesty a lower-energy path than hallucination. Cite the passages you used: passages are
numbered in the prompt, and the model marks claims with [1], [2], making every claim traceable
to a source. Low temperature (0 or near it) suits this task: grounded QA wants determinism and
precision, not creative variety.

## Prompt construction

The retrieved chunks are formatted into a numbered context block — each entry carrying its
number, source document id, and text — followed by the question. Numbering serves the citation
scheme; including the source id lets the model mention where information came from. Order
matters: models attend most reliably to the beginning of the context (and suffer "lost in the
middle" degradation on very long contexts), so the retriever's ranking should be preserved —
best passage first, and a context budget that stays well inside the model's window.

## Citations and verification

Citations turn an answer from an assertion into an argument: a reader (or an automated checker)
can open passage [2] and verify the claim against it. A subtle implementation point is that the
citations attached to an answer should reflect the passages the model actually cited in its
text, not simply every passage that was retrieved — parsing the [n] markers out of the answer
is the honest version. Stronger systems go further: a verification pass checks each cited
sentence against its cited passage and flags unsupported claims before the user sees them.

## Failure modes

Grounded generation fails in predictable ways worth testing deliberately. Ignoring the leash:
the model answers from parametric memory when retrieval returned weak passages — caught by
asking questions whose answers are absent from the corpus and checking for refusal. Citation
laundering: correct-looking citations attached to claims the passage does not support.
Retrieval poisoning: instructions embedded inside a retrieved document ("ignore your previous
instructions...") that the model obeys as if they were the user's — a prompt-injection channel
unique to RAG, mitigated by treating retrieved text as data and never as instructions. Each of
these is invisible in casual testing and obvious the moment you probe for it.

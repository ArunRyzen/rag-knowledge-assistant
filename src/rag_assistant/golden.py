"""The golden evaluation set for the textbook corpus in `data/`.

The corpus is the Tamil Nadu State Board (Samacheer Kalvi) Class 1 English book, Term 1, split
into one file per unit. Doc ids are the filenames without extension:

    unit-1-my-pet  ·  unit-2-play-time  ·  unit-3-families

Every question below is answered by the book itself — story facts a child would be quizzed on.
Labels are keyed to UNITS (documents), not chunks, so the set survives any change to chunking
parameters. To grow the set, append more dicts in the same shape (that's a task in
`tasks/README.md`).
"""

from __future__ import annotations

GOLDEN: list[dict[str, object]] = [
    # Unit 1 — My Pet (Valli and her pet goat Chittu; Alphabet Jungle a–i; counting)
    {"question": "Who is Valli's pet?", "relevant_doc_ids": ["unit-1-my-pet"]},
    {"question": "What does Chittu eat?", "relevant_doc_ids": ["unit-1-my-pet"]},
    {
        "question": "Where is the bear in the Alphabet Jungle story?",
        "relevant_doc_ids": ["unit-1-my-pet"],
    },
    {
        "question": "How many bees do Valli and Chittu see near the farm?",
        "relevant_doc_ids": ["unit-1-my-pet"],
    },
    # Unit 2 — Play Time (Come Let us Play: the rat wants to join the games)
    {"question": "What does the rat build in the story?", "relevant_doc_ids": ["unit-2-play-time"]},
    {
        "question": "Who is flying a kite when the rat asks to join?",
        "relevant_doc_ids": ["unit-2-play-time"],
    },
    {
        "question": "Which animals are playing cricket?",
        "relevant_doc_ids": ["unit-2-play-time"],
    },
    {
        "question": "What are the monkeys doing when the rat wants to play with them?",
        "relevant_doc_ids": ["unit-2-play-time"],
    },
    # Unit 3 — Families (Nila's family; My Family and Friends)
    {
        "question": "What does Nila call her father and her mother?",
        "relevant_doc_ids": ["unit-3-families"],
    },
    {"question": "Where does Nila live?", "relevant_doc_ids": ["unit-3-families"]},
    {"question": "What is Nila's sister's name?", "relevant_doc_ids": ["unit-3-families"]},
    {
        "question": "What jobs do Abdul's mother and father do?",
        "relevant_doc_ids": ["unit-3-families"],
    },
]

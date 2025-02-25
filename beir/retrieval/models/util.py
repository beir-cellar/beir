from __future__ import annotations


def extract_corpus_sentences(corpus: list[dict[str, str]] | dict[str, list] | list[str], sep: str) -> list[str]:
    """Extracts sentences from the corpus"""
    if isinstance(corpus, dict):
        sentences = [
            (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()
            for i in range(len(corpus["text"]))
        ]

    elif isinstance(corpus, list):
        if isinstance(corpus[0], str):  # if corpus is a list of strings
            sentences = corpus
        else:
            sentences = [
                (doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus
            ]
    return sentences

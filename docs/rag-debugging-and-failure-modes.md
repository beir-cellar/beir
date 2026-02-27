# RAG debugging checklist: 16 failure modes

This page gives a small, optional debugging checklist for teams that use BEIR-style retrieval as part of a RAG pipeline.

The goal is simple. Benchmarks can tell you whether a retriever behaves reasonably. Once you connect that retriever to an LLM and a real product, new kinds of failures appear. This page gives you a short vocabulary for naming those failures.

It does not prescribe any specific framework or solution. It is meant to be a neutral map you can layer on top of your own stack.

We use the following tags:

- `[IN]` input and retrieval  
- `[RE]` reasoning and planning  
- `[ST]` state and context  
- `[OP]` infra and deployment  
- `{OBS}` observability and evaluation  
- `{SEC}` security  
- `{LOC}` language and OCR  

---

## 1. Quick map: sixteen failure modes

Use this as a one-page index when you review incidents.

| # | problem domain (with layer / tags) | what breaks |
| --- | --- | --- |
| 1 | [IN] hallucination & chunk drift {OBS} | retrieval returns wrong or irrelevant content |
| 2 | [RE] interpretation collapse | chunk is right, logic is wrong |
| 3 | [RE] long reasoning chains {OBS} | drifts across multi-step tasks |
| 4 | [RE] bluffing / overconfidence | confident but unfounded answers |
| 5 | [IN] semantic ≠ embedding {OBS} | cosine match ≠ true meaning |
| 6 | [RE] logic collapse & recovery {OBS} | dead-ends, needs controlled reset |
| 7 | [ST] memory breaks across sessions | lost threads, no continuity |
| 8 | [IN] debugging is a black box {OBS} | no visibility into failure path |
| 9 | [ST] entropy collapse | attention melts, incoherent output |
| 10 | [RE] creative freeze | flat, literal outputs |
| 11 | [RE] symbolic collapse | abstract or logical prompts break |
| 12 | [RE] philosophical recursion | self-reference loops, paradox traps |
| 13 | [ST] multi-agent chaos {OBS} | agents overwrite or misalign logic |
| 14 | [OP] bootstrap ordering | services fire before dependencies are ready |
| 15 | [OP] deployment deadlock | circular waits inside infra |
| 16 | [OP] pre-deploy collapse {OBS} | version skew or missing secret on first call |

---

## 2. How BEIR users can apply this

A simple workflow that respects how BEIR is already used in practice:

1. **Run BEIR as usual**  
   Use the existing benchmarks and metrics to validate your retriever choices.

2. **Collect concrete failure examples**  
   Once your RAG system is live, keep a short log of questions where the answer is clearly wrong, unstable, or missing, even though retrieval looked OK.

3. **Map each incident to one row**  
   For each failure, ask “which row in the table does this most resemble”.  
   You do not need a perfect match. The point is to have a rough label.

4. **Group by failure mode**  
   When many incidents share the same row, they usually want the same kind of fix.  
   - lots of No.1 or No.5 → rethink chunking, query rewriting, embeddings, or index hygiene  
   - lots of No.2, No.3 or No.6 → adjust prompts, reasoning depth, or how you post-process retrieved passages  
   - lots of No.8 → improve logging, attributions and retrieval inspection tools  

5. **Re-run BEIR and re-check incidents**  
   After changes, re-run BEIR benchmarks as usual, and also check whether your real incidents are moving from one row to another or actually disappearing.

This way, BEIR keeps its role as the core retrieval benchmark, and this checklist only acts as a thin layer for failure analysis around it.

---

## 3. Origin

This checklist is adapted from an open-source “16 problem map” for RAG and agent pipelines (MIT-licensed), maintained in the [WFGY 16 Problem Map](https://github.com/onestardao/WFGY/blob/main/ProblemMap/README.md)  

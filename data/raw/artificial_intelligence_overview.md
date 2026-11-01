# Artificial Intelligence and Generative AI Overview

## Artificial Intelligence, Machine Learning, and Deep Learning

Artificial Intelligence (AI) is the broad field of building systems that perform tasks normally
requiring human intelligence. Machine Learning (ML) is a subset of AI focused on systems that learn
patterns from data. Deep Learning is a subset of Machine Learning that uses multi-layered neural
networks to learn hierarchical representations of data, and is particularly effective for
unstructured data such as images, audio, and text.

## Neural Networks

A neural network is composed of layers of interconnected nodes ("neurons"), each applying a
weighted sum of its inputs followed by a non-linear activation function (such as ReLU or sigmoid).
During training, an algorithm called backpropagation adjusts the weights to minimize a loss
function, typically using gradient descent or a variant such as Adam.

## Large Language Models (LLMs)

Large Language Models are neural networks, typically based on the Transformer architecture,
trained on massive amounts of text to predict the next token in a sequence. This simple training
objective, at sufficient scale, results in models capable of a wide range of language tasks:
answering questions, summarizing, translating, and reasoning about text.

## Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a retrieval system with a language model: instead of
relying solely on knowledge memorized during training, the system first retrieves relevant
passages from an external knowledge base and includes them in the prompt given to the language
model. This grounds the model's answer in specific, verifiable source material, reduces
hallucination, and allows the system to answer questions about information the model was never
trained on (such as private documents or recent events).

## Embeddings

An embedding is a dense numerical vector representation of text (or another data type) such that
semantically similar items are close together in the vector space. Embeddings are the foundation of
semantic search: a query is embedded into the same vector space as the documents, and the documents
with the closest vectors (by cosine similarity or another distance metric) are considered most
relevant to the query.

## Hallucination

Hallucination refers to a language model generating text that is fluent and confident-sounding but
factually incorrect or unsupported by any real source. RAG systems reduce hallucination by grounding
answers in retrieved evidence, but well-designed systems must still detect when the retrieved
evidence is insufficient and explicitly say so rather than guessing.

## Prompt Engineering

Prompt engineering is the practice of designing the instructions and context given to a language
model to reliably produce the desired output. Effective techniques include giving clear,
unambiguous instructions, providing examples (few-shot prompting), asking the model to reason
step by step, and specifying the desired output format explicitly.

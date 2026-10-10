"""
Answer prompts. They tell the model to answer only from the retrieved chunks
and to say so, with a fixed sentence, when the chunks do not contain the answer.
"""
from llama_index.core import PromptTemplate

# The exact reply for "not in the documents"; the app and the evaluation detect it by this text.
NO_ANSWER = "I don't know based on the provided documents."

TEXT_QA_PROMPT = PromptTemplate(
    "Context information from the user's documents is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the query using only the context above, not prior knowledge.\n"
    "If the context answers the query, even partly or in different words, answer from it.\n"
    f'Only if the context has nothing that answers the query, reply exactly: "{NO_ANSWER}"\n'
    "Query: {query_str}\n"
    "Answer: "
)

REFINE_PROMPT = PromptTemplate(
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer (only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Using only this context and the existing answer, refine the answer to better answer the query. "
    "If the context isn't useful, return the existing answer unchanged. "
    f'If neither contains the answer, reply exactly: "{NO_ANSWER}"\n'
    "Refined Answer: "
)


def is_no_answer(text: str) -> bool:
    """Whether a response is the fixed "not in the documents" reply."""
    return NO_ANSWER.lower().rstrip(".") in text.strip().lower()

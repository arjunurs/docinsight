"""
Answer prompts. They tell the model to answer only from the retrieved chunks
and to say so, with a fixed sentence, when the chunks do not contain the answer.
"""
from llama_index.core import PromptTemplate

# The exact reply for "not in the documents"; the app and the evaluation detect it by this text.
NO_ANSWER = "I don't know based on the provided documents."

# Rules shared by both prompts. They name answers the model has to read off the
# context rather than quote (a flattened table row, the latest item in a list, a
# fact stated as of the document's date). On SQuAD this cut wrong declines from 3
# to 1 of 100. On Natural Questions 7 of 100 are still declined: each gold chunk
# answers only indirectly (it never names the film or the show asked about), and
# gpt-4o-mini declines those whenever the prompt offers a decline. Answering them
# took a prompt with no decline plus a second LLM call to detect free-text
# declines, which doubles the calls and added wrong answers, so it was not shipped.
# See "When the app says it doesn't know" in the README.
_ANSWER_RULES = (
    "Answer the query using only the context above, not prior knowledge.\n"
    "Queries are often short search phrases (\"actor who plays X\", \"last time Y won\"); "
    "read them for their plain meaning.\n"
    "The context counts as answering the query when the answer can be read from it, even if:\n"
    "- it is worded differently from the query or answers only part of it;\n"
    "- it sits in a table or list (rows may be flattened onto one line, cells separated by spaces);\n"
    "- it takes picking an item (the latest, first, largest) or counting entries in the context;\n"
    "- it is stated as of the document's date (\"currently\", \"latest\"): give it and say so.\n"
    "If the query depends on today's date (an age, \"how long ago\"), give the dates the context "
    "states instead of calculating from today.\n"
)

TEXT_QA_PROMPT = PromptTemplate(
    "Context information from the user's documents is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    + _ANSWER_RULES
    + f'If nothing in the context answers the query, reply exactly: "{NO_ANSWER}"\n'
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
    + _ANSWER_RULES
    + "Refine the existing answer with this context only if it helps. "
    "If the context isn't useful, return the existing answer unchanged. "
    f'If neither the existing answer nor the context answers the query, reply exactly: "{NO_ANSWER}"\n'
    "Refined Answer: "
)


def _normalize(text: str) -> str:
    return text.strip().strip("\"'").strip().rstrip(".").replace("\u2019", "'").lower()


def is_no_answer(text: str) -> bool:
    """
    Whether a response is exactly the fixed "not in the documents" reply, ignoring
    case, surrounding quotes and a trailing period. A reply that adds anything
    else ("I don't know based on the provided documents, but ...") is an answer.
    """
    return _normalize(text) == _normalize(NO_ANSWER)

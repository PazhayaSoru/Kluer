from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate


extract_prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "You are extracting relevant entities like organization, person, food etc from the text",
    ),
    ("system",
     "Use the given format to extract information from the following"
     "input: {question}"),
  ]
)

condense_question_template = """
Given the following conversation and a follow up question, rephrase the follow up question
to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

base_template = """ Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.

Answer:

"""

base_prompt = ChatPromptTemplate.from_template(base_template)



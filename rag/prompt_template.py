""" Custom prompt template
"""
from langchain_core.prompts import PromptTemplate

TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""

PROMPT = PromptTemplate.from_template(TEMPLATE)

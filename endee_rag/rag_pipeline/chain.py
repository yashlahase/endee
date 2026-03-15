import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class RAGPipeline:
    """ LangChain pipeline for Retrieval Augmented Generation using Google Gemini. """

    def __init__(self, llm_model: str = "gemini-1.5-pro"):
        # langchain-google-genai uses GOOGLE_API_KEY env var
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0)
        
        self.template = """
        Answer the question based only on the following context:
        {context}

        Question: {question}
        
        Answer should be professional and grounded in the provided context. If the context doesn't contain the answer, state that you don't have enough information.
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.output_parser = StrOutputParser()

    def format_docs(self, docs: List[dict]) -> str:
        """ Formats retrieved document hits into a single context string. """
        context_parts = []
        for hit in docs:
            text = hit.get("meta", "")
            if text:
                context_parts.append(text)
        
        return "\n\n---\n\n".join(context_parts)

    def run(self, query: str, context_docs: List[dict]) -> str:
        """ Direct execution. """
        context = self.format_docs(context_docs)
        formatted_prompt = self.template.format(context=context, question=query)
        response = self.llm.invoke(formatted_prompt)
        return response.content

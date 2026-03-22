from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI



load_dotenv()

model_translation = ChatGroq(model= "llama3-8b-8192")
# model_summarisation = ChatGroq(model = "llama-3.3-70b-versatile")
# model_translation = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt_lang_check = ChatPromptTemplate.from_messages([
    ("system","You are helpful assistant that detects the language of given text"),
    ("user","Find the language of given text {question}. Only provide one word answer of respective language. Do not provide any extra text apart from language")
])

parser= StrOutputParser()

def is_english(text):
    if text=="English":
        return True
    else:
        return False
    

# lang_check_chain = prompt_lang_check | model_translation | parser | RunnableLambda(is_english)
lang_check_chain = prompt_lang_check | model_translation | parser

result = lang_check_chain.invoke({"question":"mi mazya mulana changle sanskar kase deu shakto?"})
print(result)
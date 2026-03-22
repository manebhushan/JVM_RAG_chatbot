from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Pinecone
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from itertools import islice
# from langdetect import detect, DetectorFactory
import re
from langchain.prompts import PromptTemplate


load_dotenv()

model_translation = ChatGroq(model= "llama3-8b-8192", temperature=0.0)
model_summarisation = ChatGroq(model = "llama-3.3-70b-versatile",temperature=0.5)

prompt_translation_mr_to_en = ChatPromptTemplate.from_messages([
    ("system","You are helpful assistant that translates Marathi text to English"),
    ("user","Translate the following Marathi text {question} to English. Do not provide any extra text apart from translation")
])

parser= StrOutputParser()

mr_to_en_chain = prompt_translation_mr_to_en | model_translation | parser

# text_mr = "नमस्कार, तुम्ही कसे आहात?"
# translation = mr_to_en_chain.invoke({"text":text_mr})

# print(translation)

embeddings = OllamaEmbeddings(model="sam860/granite-embedding-english:125m-Q8_0")
faiss_db = FAISS.load_local("guidance_to_students_faiss_V4", embeddings, allow_dangerous_deserialization=True)
# # # for printing 10th page when docs is created using lazy_load as docs then becomes a generator
# # doc_in_between = next(islice(docs,10,11))
# # print(doc_in_between.page_content)

retriever = faiss_db.as_retriever(search_type = "similarity",kwargs={"k":4})

# # search_result = retriever.invoke("how to score good marks in exam?")

# # for result in search_result:
# #     print(f"\n for ducument id {result.id} page content is: \n{result.page_content}")

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
    

lang_check_chain = prompt_lang_check | model_translation | parser | RunnableLambda(is_english)




# question= "आपण कसे आहात?"
# question = "परीक्षेत चांगले गुण कसे मिळवावेत?"
# question = "लहान वयातल्या प्रेमाबद्दल तुम्ही आम्हाला मार्गदर्शन करू शकता का?"
# question = "how to stay away from addictions"
# question = "जीवंनविद्या विद्यार्थी दशेत कशी लागू करावी?"
# question = "how to improve focus while studying?"
# question = "what is the purpose of education?"
# question = "What role does luck play in life?"
# question= "what is destiny?"
# question = "I get good marks in one subject but not in other subjects. Why this could be and how can I improve my performance in other subjects?"
# question = "does praying to god before exam will get me good marks?"
question ="what are your thoughts on students watching tv?"

direct_chain = RunnableLambda(lambda x:x["question"])
# direct_chain = RunnablePassthrough({"question":question})
question_chain = RunnableBranch(
    (lang_check_chain, direct_chain ),
    mr_to_en_chain
)

# question_chain = RunnableBranch(
#     (
#         lambda x: (
#             print("[DEBUG] inside branch condition:", x, is_english(x["question"])) or is_english(x["question"])
#         ),
#         direct_chain
#     ),
#     mr_to_en_chain
# )

# trans = question_chain.invoke({"text":question})

# print(trans)
# print(question_chain.invoke({"question": "does praying to god before exam will get me good marks?"}))
# print(question_chain.invoke({"text": "does praying to god before exam will get me good marks?"}))
# print(question_chain.invoke("जीवंनविद्या विद्यार्थी दशेत कशी लागू करावी?"))




# similar_chunks = context_chain.invoke({"text":question})

# print(similar_chunks)

def append_chunks(chunks):
    context=""
    for chunk in chunks:
        context = context +" "+ chunk.page_content
    return context

# append_chunks(similar_chunks)

context_chain = retriever | RunnableLambda(append_chunks)

prompt = PromptTemplate(
    template= """
        You are a helpful summarization assistant. 
        Answer the given {question} by summarising the {context}. Provide answer such that a 
        10 year old can understand. Answer only from the given context, Do not add any other lines
        from your own knowledge. 
        Always use lines like "Sadguru Wamanrao Pai says" or "According to Sadguru Wamanrao Pai" while answering, as the knowledge is given to us by him. 
""",
input_variables=["question","context"]
)

q_c_chain = question_chain | RunnableParallel(
        question = RunnablePassthrough(),
        context = context_chain
)

main_chain = q_c_chain | prompt | model_summarisation | parser

config ={
    "run_name":"with_lang_check_using_llm"
}
result = main_chain.invoke({"question":question}, config=config)
print(result)
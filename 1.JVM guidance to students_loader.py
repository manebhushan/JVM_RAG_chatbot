from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from itertools import islice
# from langdetect import detect, DetectorFactory
import re




print("Loading the document...")
loader = PyMuPDFLoader("Jeevanvidyas-guidance-to-students.pdf")
print("Splitting the document...")
splitter = RecursiveCharacterTextSplitter(chunk_size = 1200,chunk_overlap = 100)
embeddings = OllamaEmbeddings(model="sam860/granite-embedding-english:125m-Q8_0")

docs = loader.load()
# print(type(docs))
# print(docs[10:11])
docs = docs[10:]
for doc in docs[10:]:
    # Pattern 1
    if re.search(r"JEEVAN VIDYA'S GUIDANCE TO STUDENTS\n\d+\n\.{10,}\n", doc.page_content):
        doc.page_content = re.sub(r"JEEVAN VIDYA'S GUIDANCE TO STUDENTS\n\d+\n\.{10,}\n", "", doc.page_content)
        print(f"for {doc.metadata['page']} contains the text in if")
    # Pattern 2
    elif re.search(r"JEEVAN VIDYA'S GUIDANCE TO STUDENTS\n\.{10,}\n", doc.page_content):
        doc.page_content = re.sub(r"\d+\nJEEVAN VIDYA'S GUIDANCE TO STUDENTS\n\.{10,}\n", "", doc.page_content)
        print(f"for {doc.metadata['page']} contains the text in elif")
    
    doc.page_content = doc.page_content.replace('\x91', "'").replace('\x92', "'")
    doc.page_content = doc.page_content.replace('\x93', '"').replace('\x94', '"')
    doc.page_content = re.sub(r'(?<!\.)\n', ' ', doc.page_content)

print("document loading complete.......................................")

chunks = splitter.split_documents(docs)

# for chunk in chunks[-4]:
#     print(f"chunks \n{chunk}")




faiss_db = FAISS.from_documents(chunks,embeddings)
faiss_db.save_local("guidance_to_students_faiss_V4")
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
from langchain.prompts import PromptTemplate
from lingua import Language, LanguageDetectorBuilder


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





# This line is optional but can improve consistency on some systems
# DetectorFactory.seed = 0




# Build the detector once to reuse it for all calls
# This makes the function much more efficient
detector = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models() # This is the default setting
    .with_minimum_relative_distance(0.9) # A higher value increases confidence
    .build()
)

def is_english(text: str) -> bool:
    """
    Checks if a given string is in English using the lingua-py library.
    
    Args:
        text: The input string to check.

    Returns:
        True if the detected language is English, False otherwise.
    """
    # Preprocessing to handle special characters and numbers, which can confuse some detectors
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text["question"]).lower().strip()
    
    # If the text is empty or too short, it's difficult to make a reliable prediction
    if not clean_text or len(clean_text.split()) < 2:
        return False
    
    # Use the pre-built detector to detect the language
    detected_language = detector.detect_language_of(clean_text)
    
    # Return True if the detected language is English
    return detected_language == Language.ENGLISH



# def is_english(text: str) -> bool:
#     """
#     Checks if a given string is in English.

#     Args:
#         text: The input string to check.

#     Returns:
#         True if the detected language is English, False otherwise.
#     """
#     # Langdetect can struggle with very short or non-alphabetic text.
#     # We preprocess the text to ensure it's suitable for detection.
#     # Remove special characters and numbers, and convert to lowercase.
#     clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    
#     # If the cleaned text is too short, langdetect may not be reliable.
#     # We can handle this by returning False or using a different logic.
#     if len(clean_text.split()) < 3: # A simple heuristic for short text
#         return False

#     try:
#         # Detect the language of the cleaned text
#         lang_code = detect(clean_text)
        
#         # Check if the detected language code is 'en' for English
#         return lang_code == 'en'
#     except Exception:
#         # If an error occurs (e.g., text is empty or non-alphabetic),
#         # we can't reliably detect the language.
#         return False

def condition(x):
    if isinstance(x, dict):
        text = x.get("question") or x.get("text") or ""
    else:
        text = str(x)
    return is_english(text)


# # Example usage:
# question1 = "How are you doing today?"
# question2 = "आपण कसे आहात?"
# question3 = "Hello" # A short, simple word
# question4 = "1234567890" # Non-alphabetic

# print(f"'{question1}' is English: {is_english(question1)}")
# print(f"'{question2}' is English: {is_english(question2)}")
# print(f"'{question3}' is English: {is_english(question3)}")
# print(f"'{question4}' is English: {is_english(question4)}")

# question= "आपण कसे आहात?"
# question = "परीक्षेत चांगले गुण कसे मिळवावेत?"
# question = "लहान वयातल्या प्रेमाबद्दल तुम्ही आम्हाला मार्गदर्शन करू शकता का?"
question = "how to stay away from addictions"
# question = "जीवंनविद्या विद्यार्थी दशेत कशी लागू करावी?"
# question = "how to improve focus while studying?"
# question = "what is the purpose of education?"
# question = "What role does luck play in life?"
# question= "what is destiny?"
# question = "I get good marks in one subject but not in other subjects. Why this could be and how can I improve my performance in other subjects?"
# question = "does praying to god before exam will get me good marks?"

direct_chain = RunnableLambda(lambda x:x["question"])
# direct_chain = RunnablePassthrough({"question":question})
question_chain = RunnableBranch(
    (is_english, direct_chain ),
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

context_chain = question_chain | retriever | RunnableLambda(append_chunks)

prompt = PromptTemplate(
    template= """
        You are a helpful summarization agent. 
        Answer the given {question} by summarising the {context}. Provide answer such that a 
        10 year old can understand  
""",
input_variables=["question","context"]
)

q_c_chain = RunnableParallel(
        question = question_chain,
        context = context_chain
)

main_chain = q_c_chain | prompt | model_summarisation | parser

result = main_chain.invoke({"question":question})
print(result)
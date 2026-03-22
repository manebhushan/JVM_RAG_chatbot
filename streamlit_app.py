import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- Initialize Models and Chains ---
# We use st.cache_resource to cache heavy resources like the models and the FAISS database.
# This ensures that these components are loaded only once when the app starts,
# significantly improving performance on subsequent user interactions.

@st.cache_resource
def initialize_chains():
    """Initializes all LangChain components and returns the main chain."""
    
    # Initialize models for translation and summarization
    model_translation = ChatGroq(model="llama3-8b-8192", temperature=0.0)
    model_summarisation = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    
    parser = StrOutputParser()

    # Prompt for translating Marathi to English
    prompt_translation_mr_to_en = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates Marathi text to English"),
        ("user", "Translate the following Marathi text {question} to English. Do not provide any extra text apart from translation")
    ])

    mr_to_en_chain = prompt_translation_mr_to_en | model_translation | parser

    # Load FAISS database and create a retriever
    embeddings = OllamaEmbeddings(model="sam860/granite-embedding-english:125m-Q8_0")
    try:
        faiss_db = FAISS.load_local("guidance_to_students_faiss_V4", embeddings, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever(search_type="similarity", kwargs={"k": 4})
    except Exception as e:
        st.error(f"Error loading FAISS database. Make sure the 'guidance_to_students_faiss_V4' directory exists. {e}")
        return None

    # Prompt and chain for language detection
    prompt_lang_check = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that detects the language of given text"),
        ("user", "Find the language of given text {question}. Only provide a one-word answer of the respective language. Do not provide any extra text apart from the language.")
    ])

    def is_english(text):
        # A simple check to see if the model's output is "English"
        return text.strip().lower() == "english"

    lang_check_chain = prompt_lang_check | model_translation | parser | RunnableLambda(is_english)

    # Chain that translates the question only if it's not English
    direct_chain = RunnableLambda(lambda x: x["question"])
    question_chain = RunnableBranch(
        (lang_check_chain, direct_chain),
        mr_to_en_chain
    )

    # Helper function to append retrieved document chunks into a single string
    def append_chunks(chunks):
        context = ""
        for chunk in chunks:
            context += " " + chunk.page_content
        return context

    # Context chain to retrieve and format the relevant document content
    context_chain = retriever | RunnableLambda(append_chunks)

    # Final prompt for summarization
    prompt = PromptTemplate(
        template="""
            You are a helpful summarization assistant. 
            Answer the given {question} by summarising the {context}. Provide an answer such that a 
            10-year-old can understand. Answer only from the given context, do not add any other lines
            from your own knowledge. 
            Always use lines like "Sadguru Wamanrao Pai says" or "According to Sadguru Wamanrao Pai" while answering, as the knowledge is given to us by him. 
        """,
        input_variables=["question", "context"]
    )

    # The main chain that orchestrates the entire process
    q_c_chain = question_chain | RunnableParallel(
        question=RunnablePassthrough(),
        context=context_chain
    )
    
    main_chain = q_c_chain | prompt | model_summarisation | parser
    
    return main_chain

# --- Streamlit UI Components ---
st.set_page_config(page_title="Sadguru Wamanrao Pai Q&A", page_icon="🧘‍♂️")

st.title("Sadguru Wamanrao Pai's Guidance")
st.markdown("Ask a question about life and spirituality, and I'll provide a summarized answer based on the teachings from the book 'Guidance to Students'.")

# Initialize the main RAG chain using the cached function
main_chain = initialize_chains()

if main_chain is not None:
    # Text input for the user's question
    user_question = st.text_input(
        "Your question (in English or Marathi):",
        placeholder="e.g., How can I stay focused while studying?"
    )

    # Button to trigger the process
    if st.button("Get Guidance"):
        if user_question:
            with st.spinner("Finding the answer..."):
                try:
                    # Invoke the main chain with the user's question
                    result = main_chain.invoke({"question": user_question})
                    
                    # Display the result
                    st.subheader("Answer:")
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
        else:
            st.warning("Please enter a question to get guidance.")

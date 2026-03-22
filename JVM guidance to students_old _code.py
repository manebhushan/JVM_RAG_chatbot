from langchain_cohere import ChatCohere

from langchain_core.messages import HumanMessage, SystemMessage

# import os
# import configparser

# config = configparser.ConfigParser()
# config.read('/content/config_hf_2.ini')

# cohere = config['cohere']

# os.environ['COHERE_API_KEY'] = cohere.get('COHERE_API_KEY')

import fitz
with fitz.open("Jeevanvidyas-guidance-to-students.pdf") as doc:
    text = ""
    for page in doc:
        text += page.get_text()

print(text)

print(text.count("\n"))

text = text.replace("\n"," ")
sentence_list = text.split(".")

len(sentence_list)

import configparser
import os
# set up config parser
config = configparser.ConfigParser()
config.read("./config_hf_2.ini")  # holds secrets and keys


# load Groq config
hf = config["huggingface"]
model_id = "sentence-transformers/all-MiniLM-L6-v2"

import requests

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf.get('hf_token')}"}


def query(texts):
    response = requests.post(api_url, headers=headers,
                             json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

sentence_vectorDB = query(sentence_list)
len(sentence_vectorDB), len(sentence_vectorDB[0])

import torch
import numpy as np
sent_embeddings = torch.from_numpy(np.array(sentence_vectorDB)).to(torch.float)
sent_embeddings.shape

question = ["How to study for exams?"]
output = query(question)
query_embeddings = torch.FloatTensor(output)
from sentence_transformers.util import semantic_search

hits = semantic_search(query_embeddings, sent_embeddings, top_k=5)
selectedStory=[]
for i in range(len(hits[0])):
  print(sentence_list[hits[0][i]['corpus_id']])
  selectedStory.append(sentence_list[hits[0][i]['corpus_id']])
  str(selectedStory)

  from langchain_core.messages import HumanMessage, SystemMessage

sysMessage = SystemMessage('''Role: You are question answer bot. You need to refer to the story as a list below and answer the users question
                           in detailed manner as a paragraph.''')
sysMessage1 = SystemMessage(f'Store: {selectedStory}')


humanMessage = HumanMessage(question[0])
# max_tokens=256, temperature=0.75
# https://python.langchain.com/api_reference/cohere/chat_models/langchain_cohere.chat_models.ChatCohere.html#langchain_cohere.chat_models.ChatCohere
model = ChatCohere(model="command-r-plus",temperature=0.95)
response = model.invoke([sysMessage, sysMessage1, humanMessage])
print(response)
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_chroma import Chroma
import chromadb
import streamlit as st
from llm.bird_names import birds
import bs4

@st.cache_resource
def get_vs_retriever():

    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    main_url = "https://en.wikipedia.org/wiki/{species}"

    if os.path.exists("./chromadb/chroma.sqlite3"):
        chdb = Chroma(collection_name="birds_vs", embedding_function=embedding, persist_directory="./chromadb")
        retriever = chdb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        return retriever

    loader = WebBaseLoader(
        web_paths=[main_url.format(species=bird) for bird in birds],
        bs_kwargs={"parse_only": bs4.SoupStrainer("p")}
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    vs = Chroma(
        collection_name="birds_vs",
        embedding_function=embedding,
        persist_directory="./chromadb"
    )
    vs.add_documents(splits)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    return retriever


@st.cache_resource
def get_ai_model():
    model = GoogleGenerativeAI(model="gemini-1.5-flash", safety_settings = {
       HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
    })
    return model

load_dotenv()
retriever = get_vs_retriever()

model = get_ai_model()


initial_prompt = """

You are a bot designed to provide information about birds.
So, you should mindfully answer every question asked by the user.
To assist you in this task, for every question asked, you will also be provided with related information about the query.
You should use this information and your prior knowledge to answer the user's question.
But do not let user know that you have been provided with this information.
If user is asking a brief summary about a bird, you should follow the markdown format below:

-EXAMPLE FORMAT BEGINS HERE-
**Species-Name**: House Crow \n
**Family-Name**: Corvidae \n
**Order-Name**: Passeriformes \n
**Behaviour**: House Crows are known for their intelligence and adaptability. \n
**Food-Habits**: They are omnivorous and feed on a variety of food items. \n
**Breeding**: They build their nests in trees and buildings. \n
**Type of Call**: Crow in the audio is singing a mating call. \n 
-EXAMPLE FORMAT ENDS HERE- 

The input for you will be provided as:
Question: <question>
Context: <context>

You should output the answer to the question asked by the user.
Also encourage the user to ask more questions if they have any.
You might be also provided with 'Type of Call' information in the context.
The different 'Type of Call' values are: "Call", "Song", "Dawn Song", "Non Vocal Song", "Duet", "Flight Song", "Flight Call".
Once again, please only include 'Type of Call' if you are provided about call type information.
"""

ques_ans_format = """
Question: {question}
Context: {context}
"""

def get_llm_response_as_gen(i, question):
    extern_data = retriever.invoke(question)
    context = "".join([doc.page_content for doc in extern_data])
    final_prompt = ques_ans_format.format(question=question, context=context)
    st.session_state["history"][i][0].append(HumanMessage(final_prompt))
    response = model.stream(st.session_state["history"][i][0])
    total_text = ""
    for chunk in response:
        total_text += chunk
        yield chunk
    st.session_state["history"][i][0].pop()

    return total_text

def get_llm_response_as_text(i, question):
    extern_data = retriever.invoke(question)
    context = "".join([doc.page_content for doc in extern_data])
    final_prompt = ques_ans_format.format(question=question, context=context)
    st.session_state["history"][i][0].append(HumanMessage(final_prompt))

    response = model.invoke(st.session_state["history"][i][0])
    st.session_state["history"][i][0].pop()
    return response


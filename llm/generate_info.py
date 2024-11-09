import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

model = GoogleGenerativeAI(model="gemini-1.5-flash")

initial_prompt = PromptTemplate.from_template("""
    Write brief introduction about {species} from the knowledge you have. 
    Please use the following format.
    *Species-Name*:
    *Family-Name*:
    *Order-Name*:
    *Peculiarities*:
    *Food-Habits*:
    *Where-it-is-found*:
    """)

def get_llm_response_as_gen(i):
    response = model.stream(st.session_state["history"][i][0])
    total_text = ""
    for chunk in response:
        total_text += chunk
        yield chunk
    st.session_state["history"][i][0].append(AIMessage(total_text))

def get_llm_response_as_text(i):
    response = model.invoke(st.session_state["history"][i][0])
    st.session_state["history"][i][0].append(AIMessage(response))
    return response

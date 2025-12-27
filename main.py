import streamlit as st
from PIL import Image
import base64

# Written by Shankar ----------------------------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def on_file_upload():  # Function written by Nagasai
    st.session_state["file_uploaded"] = st.session_state["file_widget"]
    st.session_state["model_type"] = st.session_state["model_selectbox"]


# Written by Nagasai -------------------------------------------------
# Declaring some keys in the session state
if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = None
# History is list, each item corresponds to a unique chat
# Each item has two lists,
# The first list has the chat history
# Second list has format and resources i.e image/audio
if "history" not in st.session_state:
    st.session_state["history"] = []
if "show_chat" not in st.session_state:
    st.session_state["show_chat"] = -1
if "last_chat" not in st.session_state:
    st.session_state["last_chat"] = -1
if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = []
if "model_type" not in st.session_state:
    st.session_state["model_type"] = "Bird Image"

# Defining the pages
pages = {
    "home": st.Page("./tabs/home.py", title="Home"),
    "au": st.Page("./tabs/about_us.py", title="About Us"),
    "result": st.Page("./tabs/result.py", title="Result")
}

page = st.navigation(list(pages.values()), position="hidden")

# Visual elements
with st.sidebar:
    st.page_link(pages["home"])
    st.page_link(pages["au"])

    st.divider()
    with st.form(key="model select"):
        st.selectbox("Select Model", [
            "Bird Image",
            "Bird Audio",
            "Feather Image",
            "Leaf Image",
            "Trunk Image"
        ], key="model_selectbox", placeholder=st.session_state["model_type"])
        file = st.file_uploader("Upload Image/Audio", key="file_widget")
        st.form_submit_button("Predict!", on_click=on_file_upload)
    st.divider()

    # History
    hist = st.expander("History", expanded=True)

    # Render the history
    for i, chat in enumerate(st.session_state["history"]):
        if hist.button(st.session_state["chat_names"][i]):
            st.session_state["show_chat"] = i

# Checking which path to take now
if st.session_state["file_uploaded"] or st.session_state["show_chat"] != -1:
    if page != pages['result']:
        st.switch_page(pages["result"])
page.run()
# End of Nagasai's code ----------------------------------------------

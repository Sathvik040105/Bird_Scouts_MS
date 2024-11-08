import streamlit as st

# Below statement is only for debugging purposes
# st.write("Written from main.py")


# Declaring some keys in the session state
if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "show_chat" not in st.session_state:
    st.session_state["show_chat"] = -1
if "last_chat" not in st.session_state:
    st.session_state["last_chat"] = -1


# Defining the pages
pages = {
    "home": st.Page("./pages/home.py", title="Home"),
    "au": st.Page("./pages/about_us.py", title="About Us"),
    "hiw": st.Page("./pages/how_it_works.py", title="How It Works"),
    "result": st.Page("./pages/result.py", title="Result"),
}

page = st.navigation(list(pages.values()), position="hidden")

# Visual elements
with st.sidebar:
    st.page_link(pages["home"])
    st.page_link(pages["au"])
    st.page_link(pages["hiw"])
    
    st.divider()
    file = st.file_uploader("Image/Audio file uploader")
    if file: 
        st.session_state["file_uploaded"] = file
    st.divider()

    # History
    hist = st.expander("History", expanded=True)

    # Render the history
    for i, chat in enumerate(st.session_state["history"]):
        if hist.button(f"Chat {i}"):
            st.session_state["show_chat"] = i


# Checking which path to take now
if st.session_state["file_uploaded"] or st.session_state["show_chat"] != -1:
    if page != pages['result']:
        st.switch_page(pages["result"])
page.run()

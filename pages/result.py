import streamlit as st
from PIL import Image
from image.species_from_image import get_species_from_image
from llm.generate_info import get_llm_response

# Function to handle user prompt
def user_submits_prompt():
    """
    Function to handle the user prompt submission.
    """
    user_prompt = st.session_state["user_prompt"]
    i = st.session_state["show_chat"] = st.session_state["last_chat"]
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("bot"):
        information = st.write_stream(get_llm_response(user_prompt))
    st.session_state["history"][i].append({"type": "user", "convo": user_prompt})
    st.session_state["history"][i].append({"type": "bot", "convo": information})


# Below statement is only for debugging purposes
st.write("written from result.py")


# Checking if the user wants to see a previous chat
if st.session_state.get("show_chat", -1) != -1:
    i = st.session_state["show_chat"]
    st.session_state["show_chat"] = -1
    st.session_state["last_chat"] = i

    for chat in st.session_state["history"][i]:
        with st.chat_message(chat["type"]):

            if isinstance(chat["convo"], Image.Image):
                # Centering the image
                _, center_col, _ = st.columns([1, 2, 1])
                center_col.write(chat["convo"])
            else:
                st.write(chat["convo"])

# If user ended up at result page without wanting to see a previous chat
# Then he must have uploaded a file
else:
    st.session_state["last_chat"] = len(st.session_state["history"])
    img = Image.open(st.session_state["file_uploaded"])
    st.session_state["file_uploaded"] = None
    img = img.resize((300, 300))
    species = get_species_from_image(img)

    # Displaying the image sent by user
    with st.chat_message("user"):
        _, center_col, _ = st.columns([1, 2,1])
        center_col.write(img)

    # Bot response
    with st.chat_message("bot"):
        # Showing the spinning animation till the information is generated
        with st.spinner("Generating information..."):
            information = get_llm_response("Write brief introduction about " + species)
        full_response = st.write_stream(information)

    # Adding the current chat to the history
    user_chat = {"type": "user", "image": True, "convo": img}
    bot_chat = {"type": "bot", "convo": full_response}
    st.session_state["history"].append([user_chat, bot_chat])

user_prompt = st.chat_input("Ask your follow up question!", key="user_prompt", on_submit=user_submits_prompt)



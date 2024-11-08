import streamlit as st
from PIL import Image
from image.species_from_image import get_species_from_image
from llm.generate_info import get_llm_response
from MTL.mtl_species_classi import mtl_species_classi


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


def show_previous_history():
    i = st.session_state["show_chat"]
    st.session_state["show_chat"] = -1

    for chat in st.session_state["history"][i]:
        with st.chat_message(chat["type"]):

            if isinstance(chat["convo"], Image.Image):
                # Centering the image
                _, center_col, _ = st.columns([1, 2, 1])
                center_col.write(chat["convo"])
            else:
                st.write(chat["convo"])

def get_info_from_species(species):
    prefix = """
    Write brief introduction about the species mentioned below.
    Please use the following format.
    *Species-Name*:
    *Family-Name*:
    *Order-Name*:
    *Peculiarities*:
    *Food-Habits*:
    *Where-it-is-found*:
    ------
    """
    return get_llm_response(prefix + species)

def show_image_and_gen():
    img = Image.open(st.session_state["file_uploaded"])
    img = img.resize((300, 300))
    species = get_species_from_image(img)

    # Displaying the image
    with st.chat_message("user"):
        _, center_col, _  = st.columns([1, 2, 1])
        center_col.write(img)

    info = get_info_from_species(species)
    info = st.write_stream(info)

    st.session_state["history"].append([])
    st.session_state["history"][-1].append({
        "type": "user",
        "convo": img
    })
    st.session_state["history"][-1].append({
        "type": "bot",
        "convo": info
    })

def show_audio_and_gen():
    audio = None
    species = "Sparrow" 

    # Display the melspectogram
    st.write("Mel spectrogram here")

    info = get_info_from_species(species)
    info = st.write_stream(info)

    st.session_state["history"].append([])
    st.session_state["history"][-1].append({
        "type": "user",
        "convo": audio 
    })
    st.session_state["history"][-1].append({
        "type": "bot",
        "convo": info
    })



############################ PAGE LOGIC STARTS HERE ###################################



# Below statement is only for debugging purposes
st.write("written from result.py")


# Checking if the user wants to see a previous chat
if st.session_state["show_chat"] != -1:
    st.session_state["last_chat"] = st.session_state["show_chat"]
    show_previous_history()

# If user ended up at result page without wanting to see a previous chat
# Then he must have uploaded a file
# If it is image
elif st.session_state["file_uploaded"].type.find("image") != -1:
    st.session_state["last_chat"] = len(st.session_state["history"])
    with st.spinner("Generating Information..."):
        show_image_and_gen()
    st.session_state["file_uploaded"] = None

# If it is audio
elif st.session_state["file_uploaded"].type.find("audio") != -1:
    st.session_state["last_chat"] = len(st.session_state["history"])
    with st.spinner("Generation of audio"):
        show_audio_and_gen()
    st.session_state["file_uploaded"] = None

# Not audio or image
else:
    st.error("Unsupported file Type")

user_prompt = st.chat_input("Ask your follow up question!", key="user_prompt", on_submit=user_submits_prompt)



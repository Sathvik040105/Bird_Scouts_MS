# Written by Nagasai

import streamlit as st

example_variable = st.session_state.user_state.get(
    "user_name", "Default Value")

st.title(f"Welcome to Bird Scouts, {example_variable}!🐦🌿")

st.subheader("""Bird Scouts is your digital companion for bird identification. This application can identify bird species by analyzing images or audio clips of their songs and calls. Whether you're a birdwatcher, researcher, or simply curious about the feathered friends around you, Bird Scouts brings the world of avian diversity right to your fingertips.""")

st.subheader("""Simply upload a photo or an audio recording, and Bird Scouts will predict the bird species and provide insights on their unique calls. Discover, learn, and explore the fascinating world of birds with Bird Scouts!
""")

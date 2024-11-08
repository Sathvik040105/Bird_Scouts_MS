import streamlit as st

# Below statement is only for debugging purposes
# st.write("Written from how it works.py")

text = """
When user uploads Audio/Image file, the backend detects the type of file it is by using it's extension.
Depending on the type of file, file will be sent to a particular pipeline.

*Image*:
Some CNN bullshit

*Audio*:
Some audio bullshit
"""
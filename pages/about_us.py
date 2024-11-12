import streamlit as st

# Below statement is only for debugging purposes
# st.write("written from about us.py")

"""
members = ["Adithya", "Krishna", "Nagasai", "Sathvik", "Sanyat", "Shankar"]

for member in members:
    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.markdown(f"**{member}**")
        col2.markdown(f"Github:\n")
        col2.markdown(f"Email:{member}@gmail.com\n")
        col2.markdown(f"Linkedln:something")
"""

with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.markdown(f"**Aditya**")
        col2.markdown(f"Github:[Aditya-Manjunatha]{https://github.com/Aditya-Manjunatha}\n")
        col2.markdown(f"Email:aditya.manjunatha01@gmail.com\n")
        col2.markdown(f"Linkedln:[Aditya Manjunatha]{www.linkedin.com/in/aditya-manjunatha-33b620336}")

# Written by Nagasai
import streamlit as st

members = ["Aditya", "Krishna", "Nagasai", "Sanyat", "Sathvik", "Shankar"]

emails = ['maditya', 'krishnal', 'nagasaij',
          'sanyatvinod', 'sathvikm', 'shankaradith']
linkedins = [
    'https://www.linkedin.com/in/aditya-manjunatha-33b620336/',
    'https://www.linkedin.com/in/krishna-eyunni-124879264/',
    'https://www.linkedin.com/in/nagasai-jajapuram/',
    'https://www.linkedin.com/in/sanyatfale/',
    'https://www.linkedin.com/in/sathvik-manthri-365984259/',
    'https://www.linkedin.com/in/shankaradithyaa-venkateswaran-12a283262/']

githubs = [
    "https://github.com/Aditya-Manjunatha",
    "https://github.com/ELNKrishna"
    "https://github.com/Nagasai561",
    "https://github.com/SanyatFale",
    "https://github.com/Sathvik040105",
    "https://github.com/OmegaSun18"
]

for i in range(6):
    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.markdown(f"**{members[i]}**")
        col2.markdown(f"Github: {githubs[i]}\n")
        col2.markdown(f"Email: {emails[i]}@iisc.ac.in\n")
        col2.markdown(f"Linkedln: [{members[i]}]({linkedins[i]})")

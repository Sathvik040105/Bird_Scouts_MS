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
    "https://github.com/ELNKrishna",
    "https://github.com/Nagasai561",
    "https://github.com/SanyatFale",
    "https://github.com/Sathvik040105",
    "https://github.com/OmegaSun18"
]

col1, col2 = st.columns(2)
cols = [col1, col2]

for i, col in enumerate(cols):
    for j in range(3):
        with col.container(border = True):
            st.markdown(f"**{members[i*3+j]}**")
            st.markdown(f"&nbsp; &nbsp; [Github]({githubs[i*3+j]}) &nbsp; &nbsp; [Email]({emails[i*3+j]}@iisc.ac.in) &nbsp; &nbsp; [Linkedln]({linkedins[i*3+j]})")
            

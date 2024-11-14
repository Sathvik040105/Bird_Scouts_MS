# Written by Nagasai
import streamlit as st

members = ["Aditya", "Krishna", "Nagasai", "Sathvik", "Sanyat", "Shankar"]
emails = ['maditya', 'krishnal', 'nagasaij',
          'sathvikm', 'sanyatvinod', 'shankaradith']
linkedins = ['https://www.linkedin.com/in/aditya-manjunatha-33b620336/', 'https://www.linkedin.com/in/krishna-eyunni-124879264/', 'https://www.linkedin.com/in/nagasai-jajapuram/',
             'https://www.linkedin.com/in/sathvik-manthri-365984259/', 'https://www.linkedin.com/in/sanyatfale/', 'https://www.linkedin.com/in/shankaradithyaa-venkateswaran-12a283262/']
githubs = ['Adithya', 'Krishna', 'Nagasai',
           'Sathvik', 'Sanyat', 'shankaradith']

for i in range(6):
    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.markdown(f"**{members[i]}**")
        col2.markdown(f"Github: {githubs[i]}\n")
        col2.markdown(f"Email: {emails[i]}@iisc.ac.in\n")
        col2.markdown(f"Linkedln: [{members[i]}]({linkedins[i]})")

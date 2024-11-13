import streamlit as st
import sqlite3
import base64

# st.set_page_config(initial_sidebar_state='collapsed')

def on_file_upload():
    st.session_state["file_uploaded"] = st.session_state["file_widget"]
    st.session_state["model_type"] = st.session_state["model_selectbox"]

# Initialize SQLite database 
conn = sqlite3.connect('users.db')
c = conn.cursor()

c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        first_name TEXT,
        last_name TEXT,
        user_name TEXT UNIQUE,
        password TEXT
    )
''')
conn.commit()

# Function to add a new user to the database
def add_user(first_name, last_name, user_name, password):
    c.execute('''
        INSERT INTO users (first_name, last_name, user_name, password)
        VALUES (?, ?, ?, ?)
    ''', (first_name, last_name, user_name, password))
    conn.commit()

# Function to get user details from the database
def get_user(user_name):
    c.execute('SELECT * FROM users WHERE user_name = ?', (user_name,))
    return c.fetchone()

# Create user_state
if 'user_state' not in st.session_state:
    st.session_state.user_state = {
        'first_name': '',
        'last_name': '',
        'user_name': '',
        'password': '',
        'logged_in': False
    }

def add_bg_from_file(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/png;base64,{encoded_string}");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Create navigation state
if 'page' not in st.session_state:
    st.session_state.page = 'select'

# Function to navigate to a different page
def navigate(page):
    st.session_state.page = page
    st.rerun()

if not st.session_state.user_state['logged_in']:
    if st.session_state.page == 'select':
        add_bg_from_file("free-jungle-border-clip-art-9.png")
        # Center the buttons using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.write("")  # Add vertical space
            st.write("")
            st.write("")
            st.write("")  # Add vertical space
            st.write("")
            st.write("")
            st.write("")  # Add vertical space
            st.write("")
            st.write("")
            st.write("")  # Add vertical space
            st.write("")
            st.write("")
            # st.write("")  # Add vertical space
            # st.write("")
            # st.write("")
            # st.write("")  # Add vertical space
            # st.write("")
            # st.write("")
            st.title('Welcome!')
            login = st.button('Login')
            if login:
                navigate('login')
            signup = st.button('Sign Up')
            if signup:
                navigate('signup')
    # Create login form
    elif st.session_state.page == 'login':
        # add_bg_from_file("6697303051fa63258d4da4427ba167c1-crooked-tree-birds-silhouette-landscape.png")
        st.write('Enter Credentials:')
        user_name = st.text_input('User Name')
        password = st.text_input('Password', type='password')
        submit = st.button('Login')
        back = st.button('Back')

    # Check if user is logged in
        if submit:
            user = get_user(user_name)
            if user is None:
                st.error('User not found')
            else:
                if user[3] == password:  # user[3] is the password field
                    st.session_state.user_state['user_name'] = user_name
                    st.session_state.user_state['password'] = password
                    st.session_state.user_state['logged_in'] = True
                    st.session_state.user_state['first_name'] = user[0]  # user[0] is the first_name field
                    st.session_state.user_state['last_name'] = user[1]  # user[1] is the last_name field
                    st.write('Logging In')
                    st.rerun()
                else:
                    st.write('Invalid username or password')

        if back:
            navigate('select')

    elif st.session_state.page == 'signup':
        first_name = st.text_input('First Name')
        last_name = st.text_input('Last Name')
        user_name = st.text_input('Username')
        password = st.text_input('Password', type='password')
        re_password = st.text_input('Re-enter Password', type='password')
        submit = st.button('Sign Up')
        back = st.button('Back')

        if submit:
            if not user_name or not password or not re_password:
                st.error('All fields must be filled')
            elif get_user(user_name) is not None:
                st.error('User already exists')
            elif password != re_password:
                st.error('Passwords do not match')
            else:
                add_user(first_name, last_name, user_name, password)
                st.success('User registered successfully. Please login.')
                navigate('select')
    
        if back:
            navigate('select')
elif st.session_state.user_state['logged_in']:
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
        "home": st.Page("./pages/home.py", title="Home"),
        "au": st.Page("./pages/about_us.py", title="About Us"),
        "hiw": st.Page("./pages/how_it_works.py", title="Profile"),
        "result": st.Page("./pages/result.py", title="Result"),
        "neigh": st.Page("./pages/neighbourhood.py", title="Neighbourhood")
    }

    page = st.navigation(list(pages.values()), position="hidden")

    # Visual elements
    with st.sidebar:
        st.page_link(pages["home"])
        st.page_link(pages["au"])
        st.page_link(pages["hiw"])
        st.page_link(pages["neigh"])
        
        # st.divider()
        # st.selectbox("Select Model", [
        #     "Bird Image",
        #     "Bird Audio",
        #     "Feather Image"
        # ], key="model_selectbox", placeholder=st.session_state["model_type"])
        # file = st.file_uploader("Upload Image/Audio", key="file_widget", on_change=on_file_upload)
        # st.divider()

        st.divider()
        st.selectbox("Select Model", [
            "Bird Image",
            "Bird Audio",
            "Feather Image",
            "Leaf Image",
            "Trunk Image"
        ], key="model_selectbox", placeholder=st.session_state["model_type"])
        file = st.file_uploader("Upload Image/Audio", key = "file_widget", on_change = on_file_upload)
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
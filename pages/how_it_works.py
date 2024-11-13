import streamlit as st
import sqlite3

user_name = st.session_state.user_state.get("user_name", "Default Value")
first_name = st.session_state.user_state.get("first_name", "Default Value")
last_name = st.session_state.user_state.get("last_name", "Default Value")
# Below statement is only for debugging purposes
# st.write("Written from how it works.py")

# Function to navigate to a different page
def navigate(page):
    st.session_state.page = page
    st.rerun()

def change_password(user_name, current_password, new_password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE user_name = ?', (user_name,))
    stored_password = c.fetchone()[0]
    if stored_password == current_password:
        c.execute('UPDATE users SET password = ? WHERE user_name = ?', (new_password, user_name))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


st.title("Profile")

st.subheader(f"User Name: {user_name}")

st.subheader(f"First Name: {first_name}")

st.subheader(f"Last name: {last_name}")


if 'show_change_password' not in st.session_state:
    st.session_state.show_change_password = False

if st.button("Change Password"):
    st.session_state.show_change_password = not st.session_state.show_change_password

# Change Password Form
if st.session_state.show_change_password:
    st.write("## Change Password")
    current_password = st.text_input("Current Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_new_password = st.text_input("Confirm New Password", type="password")
    change_password_button = st.button("Submit")

    if change_password_button:
        if not current_password or not new_password or not confirm_new_password:
            st.error("All fields must be filled")
        elif new_password != confirm_new_password:
            st.error("New passwords do not match")
        elif current_password == new_password:
            st.error("New password is the same as old")
        else:
            if change_password(user_name, current_password, new_password):
                st.success("Password changed successfully")
                st.session_state.show_change_password = False
            else:
                st.error("Current password is incorrect")

if st.button("Sign Out"):
    # Clear user state and navigate to login page
    st.session_state.user_state = {
        'first_name': '',
        'last_name': '',
        'user_name': '',
        'password': '',
        'logged_in': False
    }

    navigate('select')
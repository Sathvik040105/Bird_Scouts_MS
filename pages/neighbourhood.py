# Written by Nagasai

import pandas as pd
import streamlit as st
import pydeck as pdk
import sqlite3
from llm.bird_names import birds

iisc_center_coords = [13.024059, 77.566855]


@st.cache_resource
def get_db_conn():
    database_path = './sqlitedb/locations.db'
    conn = sqlite3.connect(database_path, check_same_thread=False)
    return conn


def get_df(conn):
    query = 'SELECT * FROM locations'
    df = pd.read_sql_query(query, conn)
    return df

# Update the database with the values provided by the user


def update_db():
    bird = st.session_state.selectbox
    latitude = st.session_state.latitude
    longitude = st.session_state.longitude
    username = st.session_state["user_state"].user_name
    query = f"INSERT INTO locations (birdname, username, latitude, longitude) VALUES ('{
        bird}', '{username}', {latitude}, {longitude})"
    conn.execute(query)
    conn.commit()


conn = get_db_conn()
data = get_df(conn)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position="[longitude, latitude]",
    # Use the 'color' column for the color of the points
    get_color="[255, 0, 0, 150]",
    get_radius=25,
    pickable=True,
)

# Define the map's initial view
# Making sure that IISc is at the center of the map
view_state = pdk.ViewState(
    latitude=iisc_center_coords[0],
    longitude=iisc_center_coords[1],
    zoom=14,
    pitch=0,
)


# Tooltip to show the bird name and the user who spotted it
tooltip = {
    "html": "<b>{birdname}</b><br><b>By: {username}</b>",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white",
        "fontSize": "13px",
    }
}

# Render the map with Pydeck
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip
))

# Form to add a new location
with st.form("Location Form"):
    st.selectbox("Select a bird", birds, key="selectbox")
    st.number_input("Latitude", key="latitude", format="%0.4f",
                    min_value=-90.0, max_value=90.0, step=0.0001, value=13.024)
    st.number_input("Longitude", key="longitude", format="%0.4f",
                    min_value=-180.0, max_value=180.0, step=0.0001, value=77.567)
    st.form_submit_button("Submit Location info", on_click=update_db)

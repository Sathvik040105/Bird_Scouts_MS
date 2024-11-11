import pydeck as pdk
import pandas as pd
import streamlit as st
import sqlite3

@st.cache_resource
def get_db_conn():
    database_path = './sqlitedb/locations.db'
    conn = sqlite3.connect(database_path, check_same_thread=False) 
    return conn

def get_df(conn):
    query = 'SELECT * FROM locations'
    df = pd.read_sql_query(query, conn)
    return df

conn = get_db_conn()
data = get_df(conn)

# data['color'] = data['color'].apply(lambda x: [200, 0, 0, 150] if x == 'red' else [0, 200, 0, 150])

# # Define a layer
# layer = pdk.Layer(
#     "ScatterplotLayer",
#     data=data,
#     get_position="[longitude, latitude]",
#     get_color="color",
#     get_radius=25,
#     pickable=True
# )

# # Define the view
# view_state = pdk.ViewState(
#     latitude=13.024059,
#     longitude=77.566855,
#     zoom=14,
#     pitch=0,
# )

# tooltip = {
#     "html": "<b>{name}</b>",
#     "style": {
#         "backgroundColor": "steelblue",
#         "color": "white",
#         "fontSize": "12px",
#     }
# }

# # Render the map
# st.pydeck_chart(pdk.Deck(
#     layers=[layer],
#     initial_view_state=view_state,
#     tooltip=tooltip
# ))

import folium
from streamlit_folium import st_folium

mp = folium.Map(location=[13.024059, 77.566855], zoom_start=15, tiles="CartoDB Positron", height=500)
for i in range(len(data)):
    folium.Marker([data['latitude'][i], data['longitude'][i]], tooltip=data['name'][i], 
                    icon=folium.Icon(color="red" if data['color'][i] == 'red' else "green", icon="star")).add_to(mp)
res = st_folium(mp)

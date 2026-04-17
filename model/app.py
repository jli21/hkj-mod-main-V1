"""
app.py
======
Streamlit multipage dashboard entry point.
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="HKJC Racing Model",
    layout="wide",
)

pages = st.navigation([
    st.Page("pages/0_Home.py", title="Home", default=True),
    st.Page("pages/1_Raw_Data.py", title="Raw Data"),
    st.Page("pages/2_Model_Formula.py", title="Model Overview"),
    st.Page("pages/3_Diagnostics.py", title="Diagnostics"),
    st.Page("pages/4_Horse_Analysis.py", title="Horse Analysis"),
    st.Page("pages/5_Bankroll.py", title="Bankroll"),
    st.Page("pages/6_Training.py", title="Training"),
    st.Page("pages/7_Race_Inspector.py", title="Race Inspector"),
])

pages.run()

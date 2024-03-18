import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to GenProtein! 👋")

st.markdown(
    """
     GenProtein is a framework tailored specifically for GenAI based Protein Design.

**👈 Select the Use Case Demo from the sidebar** to see some capabilities
    of what GenProtein can do!!
    ### Want to learn more?
    - Jump into [documentation](https://www.nature.com/articles/s41586-023-06728-8)
"""
)
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to ProteinDesignDemo! 👋")

st.markdown(
    """
    
    ProteinDesignDemo is a demo framework tailored specifically for GenAI for Protein Design.

    **👈 Select a page from the sidebar** to see some examples
    of what ProteinDesignDemo can do!
    ### Want to learn more?
    - Jump into [documentation](https://www.nature.com/articles/s41586-023-06728-8)
"""
)
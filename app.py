import streamlit as st
import demo

st.set_page_config(
    page_title="GenAI for Protein Design",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Protein Design Driven by Chroma")

# sidebar
#st.sidebar.header("Config")

# the mapping between the function and use case
demoDict={
    "UnconditionalGenerate":demo.GenerateProteinDemo,
    "ComplexGenerate":demo.complexSampleDemo,
    "SymmetricGenerate":demo.symmetricSampleDemo,
    "ShapeConstrainedGenerate":demo.shapeSampleDemo,
    "FoldConditionGenerate":demo.foldSampleDemo,
    "SecondStructureConditionGenerate":demo.ssSampleDemo,
    "SubstructureConditionGenerate":demo.substructureSampleDemo,

}
# sidebar for demo options
selected_branch = st.sidebar.selectbox("Select the Function to Run:", list(demoDict.keys()))
style=st.sidebar.selectbox("Select Visualization Style:Can be 'stick', 'sphere', 'cross','cartoon'",('stick', 'sphere', 'cross','cartoon'),key='style')
resn=st.sidebar.selectbox("Select the Amino Acid Type to Display",
                          ('*', 'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'),key='resn')

# Exectuion
demoDict[selected_branch](style,resn)

import streamlit as st
import demo

st.set_page_config(
    page_title="Chroma Demos",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Protein Design Demos for Chroma")

# sidebar
st.sidebar.header("Demo Config")

# 创建字典映射demo
demoDict={
    "getProtein":demo.getProteinDemo,
    "complexSample":demo.complexSampleDemo,
    "symmetricSample":demo.symmetricSampleDemo,
    "shapeSample":demo.shapeSampleDemo,
    "foldSample":demo.foldSampleDemo,
    "ssSample":demo.ssSampleDemo,
    "substructureSample":demo.substructureSampleDemo,

}
# 在侧边栏中添加一个选择框，用于选择demo
selected_branch = st.sidebar.selectbox("Select demo", list(demoDict.keys()))
style=st.sidebar.selectbox("Select style:Can be 'stick', 'sphere', 'cross','cartoon'",('stick', 'sphere', 'cross','cartoon'),key='style')
resn=st.sidebar.selectbox("Select display resn:PDB resn labels:['ALA','ARG','LYS','THR','TRP','TYR','VAL']",('','ALA','ARG','LYS','THR','TRP','TYR','VAL'),key='resn')

# 执行选定分支对应的函数
demoDict[selected_branch](style,resn)

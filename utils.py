import os
import contextlib
import torch
import warnings
from tqdm import tqdm, TqdmExperimentalWarning
from functools import partialmethod
import streamlit as st
from stmol import *
from chroma.utility.api import register_key
from dotenv import load_dotenv
import locale
from chroma import Chroma, Protein, conditioners
from chroma.models import graph_classifier, procap
from chroma.models.graph_design import GraphDesign
from chroma.utility.chroma import letter_to_point_cloud, plane_split_protein
import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
tqdm.__init__ = partialmethod(tqdm.__init__, leave=False)

torch.backends.cudnn.deterministic = True

load_dotenv()
api_key = os.getenv("API_KEY")
register_key(api_key)

locale.getpreferredencoding = lambda: "UTF-8"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
with contextlib.redirect_stdout(None):
    chroma = Chroma(device=device)


def download(outputFile,newFileName,description):
    with open(outputFile, "rb") as file:
        btn = st.download_button(
                label=description,
                data=file,
                file_name=newFileName,
            )
        


def display(output,style,resn):
    # imformation
    protein=Protein.from_PDB(output,device=device)
    st.subheader("Protein Information:")
    st.write(f"Device: GPU")
    st.write(f"Protein Length: {len(protein)} residues")
    st.write(f"Structured Residue Count: {protein.length(structured=True)}")

    # display Protein sequence
    st.subheader("Protein Sequence:")
    protein_sequence = protein.sequence(format="three-letter-list")
    st.markdown(f"**Protein Sequence:** {protein_sequence}")
    st.write(protein_sequence)
    # display Protein structure
    with open(output, "r") as file:
        pdb_content = file.read()

    obj = makeobj(pdb_content,style=style,background='white')

    # using stmol for 3d visualisation of protein structure
    st.subheader("Protein Structure:")
    traj_output = output.replace(".pdb", "_trajectory.pdb")
    
    protein_newName = st.text_input("The specified file name. Default is {}.".format(output[output.rfind("/") + 1:])+"Please press [Enter] to confirm the change before download.", value=output[output.rfind("/") + 1:], key='protein_newName')
    download(output,protein_newName,"Download sample")
    traj_newName = st.text_input("The specified file name. Default is {}.".format(traj_output[traj_output.rfind("/") + 1:])+"Please press [Enter] to confirm the change before download.", value=traj_output[traj_output.rfind("/") + 1:], key='traj_newName')
    download(traj_output,traj_newName,"Download trajectory")
    if resn !='*':
        obj = render_pdb_resn(obj ,resn_lst =resn)
    showmol(obj, width=1800)
   
def render(protein, trajectories, output="./output/protein.pdb"):
    protein.to_PDB(output)
    traj_output = output.replace(".pdb", "_trajectory.pdb")
    trajectories["trajectory"].to_PDB(traj_output)



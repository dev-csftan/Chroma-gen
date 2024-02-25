# @title Setup

# @markdown [Get your API key here](https://chroma-weights.generatebiomedicines.com) and enter it below before running.

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import contextlib

api_key = "2cdade6d058b4fd1b85fa5badb501312"  # @param {type:"string"}


import torch

# torch.use_deterministic_algorithms(False)

import warnings
from tqdm import tqdm, TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, leave=False)

import streamlit as st
from stmol import *
def download(filename,description):
    with open(filename, "rb") as file:
        btn = st.download_button(
                label=description,
                data=file,
                file_name=filename
            )
import pandas as pd
def display(output,style,resn):
    # imformation
    protein=Protein.from_PDB(output)
    st.subheader("Protein Information:")
    st.write(f"Device: {protein.device}")
    st.write(f"Protein Length: {len(protein)} residues")
    st.write(f"Structured Residue Count: {protein.length(structured=True)}")

    # 显示 Protein 的序列
    st.subheader("Protein Sequence:")
    protein_sequence = protein.sequence(format="three-letter-list")
    st.markdown(f"**My List:** {protein_sequence}")
    st.write(protein_sequence)
    # 显示 Protein 的结构
    with open(output, "r") as file:
        pdb_content = file.read()

    obj = makeobj(pdb_content,style=style,background='white')

    # 使用 stmol 展示蛋白质结构
    st.subheader("Protein Structure:")
    traj_output = output.replace(".pdb", "_trajectory.pdb")
    download(output,"Download sample")
    download(traj_output,"Download trajectory")
    if resn !='':
        obj = render_pdb_resn(obj ,resn_lst =resn)
    showmol(obj, width=1800)

    


def render(protein, trajectories, output="./output/.output/protein.pdb"):
    protein.to_PDB(output)
    traj_output = output.replace(".pdb", "_trajectory.pdb")
    trajectories["trajectory"].to_PDB(traj_output)

    

import locale

locale.getpreferredencoding = lambda: "UTF-8"

from chroma import Chroma, Protein, conditioners
from chroma.models import graph_classifier, procap
from chroma.utility.api import register_key
from chroma.utility.chroma import letter_to_point_cloud, plane_split_protein

register_key(api_key)
device = "cpu"
with contextlib.redirect_stdout(None):
    chroma = Chroma(device=device)





def proteinSample(length,steps,output):
    protein, trajectories = chroma.sample(
        chain_lengths=[length], steps=steps, full_output=True,
    )
    render(protein, trajectories, output=output)
def getProteinDemo(style,resn):
    st.sidebar.title("First demo")
    st.sidebar.header("Get a protein!")
    length=st.sidebar.number_input("length",min_value=50,max_value=250,step=10,value=160,key='length')
    steps_protein=st.sidebar.number_input("steps",min_value=150,max_value=500,step=50,value=200,key='steps_protein')
    
    output="./output/protein.pdb"
    if st.sidebar.button("Run Code with Button",key="protein"):
        proteinSample(length,steps_protein,output)

    display(output,style,resn)




def complexSample(chain1_length,chain2_length,chain3_length,chain4_length,steps,output):
    protein, trajectories = chroma.sample(
        chain_lengths=[chain1_length, chain2_length, chain3_length, chain4_length],
        steps=steps,
        full_output=True,
    )
    render(protein, trajectories, output=output)
def complexSampleDemo(style,resn):
    st.sidebar.title("Second demo")
    st.sidebar.header("Get a complex")
    st.caption("Given the lengths of individual chains, Chroma can generate a complex.")
    chain1_length=st.sidebar.number_input("chain1_length,step=10",min_value=100,max_value=500,step=10,value=400,key='chain1_length')
    chain2_length=st.sidebar.number_input("chain2_length,step=10",min_value=0,max_value=200,step=10,value=100,key='chain2_length')
    chain3_length=st.sidebar.number_input("chain3_length,step=1",min_value=0,max_value=200,step=10,value=100,key='chain3_length')
    chain4_length=st.sidebar.number_input("chain4_length,step=1",min_value=0,max_value=200,step=10,value=100,key='chain4_length')
    steps_complex=st.sidebar.number_input("steps",min_value=150,max_value=500,step=50,value=200,key='steps_complex')

    output="./output/complex.pdb"
    if st.sidebar.button("Run Code with Button",key="complex"):
        complexSample(chain1_length,chain2_length,chain3_length,chain4_length,steps_complex,output)
    display(output,style,resn)



def symmetricSample(subunit_size,conditioner,output):
    symmetric_protein, trajectories = chroma.sample(
        chain_lengths=[subunit_size],
        conditioner=conditioner,
        langevin_factor=8,
        inverse_temperature=8,
        sde_func="langevin",
        potts_symmetry_order=conditioner.potts_symmetry_order,
        full_output=True,
    )
    render(symmetric_protein, trajectories, output=output)
def symmetricSampleDemo(style,resn):
    st.sidebar.title("Third demo")
    st.sidebar.header(" Symmetry")
    st.caption(" Specify the desired symmetry type and the size of a single subunit.")
    output="./output/symmetric_protein.pdb"
    symmetry_group=st.sidebar.text_input('symmetry_group:@param ["C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "D_2", "D_3", "D_4", "D_5", "D_6", "D_7", "D_8", "T", "O", "I"]',"C_7")
    subunit_size=st.sidebar.number_input("subunit_size,step=5",min_value=10,max_value=150,step=5,value=100,key='subunit_size')
    knbr=st.sidebar.number_input("knbr,step=1",min_value=1,max_value=10,step=1,value=2,key='knbr')
    conditioner = conditioners.SymmetryConditioner(
        G=symmetry_group, num_chain_neighbors=knbr
    ).to(device)
    if st.sidebar.button("Run Code with Button",key="symmetric"):
        symmetricSample(subunit_size,conditioner,output)

    display(output,style,resn)



def shapeSample(length,conditioner,output):
    shaped_protein, trajectories = chroma.sample(
        chain_lengths=[length], conditioner=conditioner, full_output=True,
    )

    render(shaped_protein, trajectories, output=output)
def shapeSampleDemo(style,resn):
    st.sidebar.title("Fourth demo")
    st.sidebar.header(" Shape")
    st.caption(" reate a protein in the shape of a desired character of arbitrary length.")

    output="./output/shaped_protein.pdb"
    character=st.sidebar.text_input('character:@param {type:"string"}','G',key='character')
    if len(character) > 1:
        character = character[:1]
        print(f"Keeping only first character ({character})!")
    length=st.sidebar.number_input('length,step=100',min_value=100,max_value=1500,step=100,value=1000,key='length_shape')
    
    if st.sidebar.button("Run Code with Button",key="shape"):
        letter_point_cloud = letter_to_point_cloud(character)
        conditioner = conditioners.ShapeConditioner(
            letter_point_cloud,
            chroma.backbone_network.noise_schedule,
            autoscale_num_residues=length,
        ).to(device)
        shapeSample(length,conditioner,output)
        
    display(output,style,resn)


def foldSample(length,conditioner,output):
    cath_conditioned_protein, trajectories = chroma.sample(
        conditioner=conditioner, chain_lengths=[length], full_output=True,
    )
    render(cath_conditioned_protein, trajectories, output=output)
def foldSampleDemo(style,resn):
    st.sidebar.title("Fifth demo")
    st.sidebar.header(" Fold")
    st.caption("Input a [CATH number](https://cathdb.info/browse) to get chain-level conditioning, e.g. `3.40.50` for a Rossmann fold or `2` for mainly beta.")

    output="./output/cath_conditioned_protein.pdb"
    CATH=st.sidebar.text_input('CATH@param {type:"string"}','3.40.50',key='CATH')
    length=st.sidebar.number_input('length,step=10',min_value=50,max_value=250,step=10,value=130,key='length_fold')

    proclass_model = graph_classifier.load_model("named:public", device=device)
    conditioner = conditioners.ProClassConditioner("cath", CATH, model=proclass_model,device=device)
    if st.sidebar.button("Run Code with Button",key="fold"):
        foldSample(length,conditioner,output)

    display(output,style,resn)


def ssSample(conditioner,SS,output):
    ss_conditioned_protein, trajectories = chroma.sample(
        steps=500, conditioner=conditioner, chain_lengths=[len(SS)], full_output=True,
    )
    render(ss_conditioned_protein, trajectories, output=output)
def ssSampleDemo(style,resn):
    st.sidebar.title("Sixth demo")
    st.sidebar.header(" Secondary structure ")
    st.caption("Enter a string to specify residue-level secondary structure conditioning: H = helix, E = strand, T = turn.")

    output="./output/ss_conditioned_protein.pdb"

    SS=st.sidebar.text_input('SS:secondary structure @param {type:"string"}',"HHHHHHHTTTHHHHHHHTTTEEEEEETTTEEEEEEEETTTTHHHHHHHH")

    proclass_model = graph_classifier.load_model("named:public", device=device)
    conditioner = conditioners.ProClassConditioner(
        "secondary_structure", SS, max_norm=None, model=proclass_model,device=device
    )

    if st.sidebar.button("Run Code with Button",key="SS"):
        ssSample(conditioner,SS,output)

    display(output,style,resn)


def substructureSample(protein,conditioner,output):
    infilled_protein, trajectories = chroma.sample(
        protein_init=protein,
        conditioner=conditioner,
        langevin_factor=4.0,
        langevin_isothermal=True,
        inverse_temperature=8.0,
        steps=500,
        sde_func="langevin",
        full_output=True,
    )
    render(infilled_protein, trajectories, output=output)
def substructureSampleDemo(style,resn):
    st.sidebar.title("Eigth demo")
    st.sidebar.header(" Substructure ")
    st.caption("Enter a PDB ID and a selection string corresponding to designable positions.")
    st.caption("Using a substructure conditioner, Chroma can design at these positions while holding the rest of the structure fixed.")
    st.caption("The default selection cuts the protein in half and fills it in.")
    st.caption("Other selections, by position or proximity, are also allowed.")

    output="./output/infilled_protein.pdb"

    pdb_id=st.sidebar.text_input("pdb_id@param ['5SV5', '6QAZ', '3BDI'] {allow-input:true}",'5SV5',key='pdb_id')

    try:
        protein = Protein.from_PDBID(pdb_id, canonicalize=True, device=device)
    except FileNotFoundError:
        print("Invalid PDB ID! Using 3BDI")
        pdb_id = "3BDI"
        protein = Protein.from_PDBID(pdb_id, canonicalize=True, device=device)

    X, C, _ = protein.to_XCS()

    selection_string=st.sidebar.text_input("selection_string: @param ['namesel infilling_selection', 'z > 16', '(resid 50) around 10'] {allow-input:true}",'namesel infilling_selection',key='selection_string')
    residues_to_design = plane_split_protein(X, C, protein, 0.5).nonzero()[:, 1].tolist()
    protein.sys.save_selection(gti=residues_to_design, selname="infilling_selection")

    try:
        conditioner = conditioners.SubstructureConditioner(
            protein, backbone_model=chroma.backbone_network, selection=selection_string,
        ).to(device)
    except Exception:
        print("Error initializing conditioner! Falling back to masking 50% of residues.")
        selection_string = "namesel infilling_selection"
        conditioner = conditioners.SubstructureConditioner(
            protein,
            backbone_model=chroma.backbone_network,
            selection=selection_string,
            rg=True
        ).to(device)

    if st.sidebar.button("Run Code with Button",key="substructure"):
        substructureSample(protein,conditioner,output)
    display(output,style,resn)



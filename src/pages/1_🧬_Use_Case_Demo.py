import os
import json
import contextlib
import torch
import warnings
from tqdm import tqdm, TqdmExperimentalWarning
from functools import partialmethod
import streamlit as st
from stmol import *
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

locale.getpreferredencoding = lambda: "UTF-8"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with contextlib.redirect_stdout(None):
    chroma = Chroma(weights_backbone='./model_weights/chroma_backbone_v1.0.pt',
                    weights_design='./model_weights/chroma_design_v1.0.pt',
                    device=device)


def download(outputFile,newFileName,description):
    with open(outputFile, "rb") as file:
        btn = st.download_button(
                label=description,
                data=file,
                file_name=newFileName,
            )
        


def display_by_stmol(output,style,resn):
    # imformation
    protein=Protein.from_PDB(output,device=device)
    st.subheader("Protein Information:")
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

st.set_page_config(
    page_title="GenAI for Protein Design Use Case",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Protein Design Driven by Diffusion Model")

def proteinSample(length,steps,output):
    protein, trajectories = chroma.sample(
        chain_lengths=[length], steps=steps, full_output=True,
    )
    render(protein, trajectories, output=output)
def GenerateProteinDemo(style,resn):
    #st.sidebar.title("Unconditional Generation")
    st.sidebar.header("Generate a Protein Backbone")
    length=st.sidebar.number_input("chain_length:The lengths of the protein chains.Default is [160],step=10.",min_value=50,max_value=250,step=10,value=160,key='length')
    steps_protein=st.sidebar.number_input("sde_steps:The number of integration steps for the SDE.Default is 200,step=50.",min_value=150,max_value=500,step=50,value=200,key='steps_protein')
    
    output="./output/protein.pdb"
    if st.sidebar.button("Run Code with Button",key="protein"):
        proteinSample(length,steps_protein,output)
    
    display_by_stmol(output,style,resn)

def complexSample(chain1_length,chain2_length,chain3_length,chain4_length,steps,output):
    protein, trajectories = chroma.sample(
        chain_lengths=[chain1_length, chain2_length, chain3_length, chain4_length],
        steps=steps,
        full_output=True,
    )
    render(protein, trajectories, output=output)
def complexSampleDemo(style,resn):
    #st.sidebar.title("Generate a Protein Complex")
    st.sidebar.header("Generate a Protein Complex")
    st.caption("Given the lengths of individual chains, It can generate a complex.")
    chain1_length=st.sidebar.number_input("chain1_length,step=10",min_value=100,max_value=500,step=10,value=400,key='chain1_length')
    chain2_length=st.sidebar.number_input("chain2_length,step=10",min_value=0,max_value=200,step=10,value=100,key='chain2_length')
    chain3_length=st.sidebar.number_input("chain3_length,step=1",min_value=0,max_value=200,step=10,value=100,key='chain3_length')
    chain4_length=st.sidebar.number_input("chain4_length,step=1",min_value=0,max_value=200,step=10,value=100,key='chain4_length')
    steps_complex=st.sidebar.number_input("sde_steps:The number of integration steps for the SDE.Default is 200,step=50.",min_value=150,max_value=500,step=50,value=200,key='steps_complex')

    output="./output/complex.pdb"
    if st.sidebar.button("Run Code with Button",key="complex"):
        complexSample(chain1_length,chain2_length,chain3_length,chain4_length,steps_complex,output)
    display_by_stmol(output,style,resn)



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
    #st.sidebar.title("Generate a Symmetric Protein Backbone")
    st.sidebar.header("Conditional Generation on Symmetry")
    st.caption(" Specify the desired symmetry type and the size of a single subunit.")
    output="./output/symmetric_protein.pdb"
    symmetry_group=st.sidebar.text_input('symmetry_group:@param ["C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "D_2", "D_3", "D_4", "D_5", "D_6", "D_7", "D_8", "T", "O", "I"]',"C_7")
    subunit_size=st.sidebar.number_input("subunit_size:the size of a single subunit.Default is 100,step=5.",min_value=10,max_value=150,step=5,value=100,key='subunit_size')
    knbr=st.sidebar.number_input("knbr,step=1",min_value=1,max_value=10,step=1,value=2,key='knbr')
    conditioner = conditioners.SymmetryConditioner(
        G=symmetry_group, num_chain_neighbors=knbr
    ).to(device)
    if st.sidebar.button("Run Code with Button",key="symmetric"):
        symmetricSample(subunit_size,conditioner,output)

    display_by_stmol(output,style,resn)



def shapeSample(length,conditioner,output):
    shaped_protein, trajectories = chroma.sample(
        chain_lengths=[length], conditioner=conditioner, full_output=True,
    )

    render(shaped_protein, trajectories, output=output)
def shapeSampleDemo(style,resn):
    #st.sidebar.title("Generate a Shapped Protein Backbone")
    st.sidebar.header("Conditional Generation on Shape")
    st.caption("create a protein in the shape of a desired character of arbitrary length.")

    output="./output/shaped_protein.pdb"
    character=st.sidebar.text_input('character:a desired character for the shape of protein. @param {type:"string"}','G',key='character')
    if len(character) > 1:
        character = character[:1]
        print(f"Keeping only first character ({character})!")
    length=st.sidebar.number_input('chain_length:The lengths of the protein chains.Default is 500,step=100.',min_value=100,max_value=1500,step=100,value=500,key='length_shape')
    
    if st.sidebar.button("Run Code with Button",key="shape"):
        letter_point_cloud = letter_to_point_cloud(character)
        conditioner = conditioners.ShapeConditioner(
            letter_point_cloud,
            chroma.backbone_network.noise_schedule,
            autoscale_num_residues=length,
        ).to(device)
        shapeSample(length,conditioner,output)
        
    display_by_stmol(output,style,resn)


def foldSample(length,conditioner,output):
    cath_conditioned_protein, trajectories = chroma.sample(
        conditioner=conditioner, chain_lengths=[length], full_output=True,
    )
    render(cath_conditioned_protein, trajectories, output=output)
def foldSampleDemo(style,resn):
    #st.sidebar.title("Generate a Chain-level Conditioned Protein")
    st.sidebar.header("Conditional Generation on Chain-level Properties")
    st.caption("Input a [CATH number](https://cathdb.info/browse) to get chain-level conditioning, e.g. `3.40.50` for a Rossmann fold or `2` for mainly beta.")

    output="./output/cath_conditioned_protein.pdb"
    CATH=st.sidebar.text_input('CATH:protein domain annotations from <https://www.cathdb.info/>. Annotation examples include 2, 2.40, 2.40.155.','3.40.50',key='CATH')
    length=st.sidebar.number_input('chain_length:The lengths of the protein chains.Default is 130,step=10.',min_value=50,max_value=250,step=10,value=130,key='length_fold')

    proclass_model = graph_classifier.load_model("named:public", device=device)
    conditioner = conditioners.ProClassConditioner("cath", CATH, model=proclass_model,device=device)
    if st.sidebar.button("Run Code with Button",key="fold"):
        foldSample(length,conditioner,output)

    display_by_stmol(output,style,resn)


def ssSample(conditioner,SS,output):
    ss_conditioned_protein, trajectories = chroma.sample(
        steps=500, conditioner=conditioner, chain_lengths=[len(SS)], full_output=True,
    )
    render(ss_conditioned_protein, trajectories, output=output)
def ssSampleDemo(style,resn):
    #st.sidebar.title("Generate a Secondary Structure Conditioned Protein")
    st.sidebar.header(" Conditional Generation on Secondary Structure Properties")
    st.caption("Enter a string to specify residue-level secondary structure conditioning: H = helix, E = strand, T = turn.")

    output="./output/ss_conditioned_protein.pdb"

    SS=st.sidebar.text_input('secondary structure @param {type:"string"}',"HHHHHHHTTTHHHHHHHTTTEEEEEETTTEEEEEEEETTTTHHHHHHHH")

    proclass_model = graph_classifier.load_model("named:public", device=device)
    conditioner = conditioners.ProClassConditioner(
        "secondary_structure", SS, max_norm=None, model=proclass_model,device=device
    )

    if st.sidebar.button("Run Code with Button",key="SS"):
        ssSample(conditioner,SS,output)

    display_by_stmol(output,style,resn)


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
    st.sidebar.title("Generate a Sub-Structure Conditioned Protein")
    #st.sidebar.header(" Conditional Generation on Substructure Properties")
    st.caption("Enter a PDB ID and a selection string corresponding to designable positions.")
    st.caption("Using a substructure conditioner, It can design at these positions while holding the rest of the structure fixed.")
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
    display_by_stmol(output,style,resn)

def natureLanguageSample(conditioner,output):
    caption_conditioned_protein,trajectories = chroma.sample(steps=200, chain_lengths=[110], conditioner=conditioner)
    render(caption_conditioned_protein, trajectories, output=output)

def natureLanguageSampleDemo(style,resn):
    st.sidebar.title("Generate a Caption Guided Protein")
    st.caption("Here, we demonstrate backbone generation conditioned on natural language prompts.")
    st.caption(" The sampling is guided by the gradients of a structure to text model.")
    CAPTION=st.sidebar.text_input('a caption:natural language prompts.',value='Crystal structure of SH2 domain',key='caption')
    torch.manual_seed(0)
    conditioner = conditioners.ProCapConditioner(CAPTION, -1).to(device)
    output='./output/caption_conditioned_protein.pdb'
    if st.sidebar.button("Run Code with Button",key="substructure"):
        natureLanguageSample(conditioner,output)
    
    display_by_stmol(output,style,resn)

# Combining Symmetry and Secondary Structure
def cSSStructureSample(composedConditioner,output):
    symm_beta_protein,trajectories = chroma.sample(chain_lengths=[100],
        conditioner=composedConditioner,
        langevin_factor=8,
        inverse_temperature=8,
        sde_func="langevin",
        steps=500,full_output=True,)
    render(symm_beta_protein,trajectories,output=output)

def cSSStructureSampleDemo(style,resn):
    st.sidebar.title("Generate a Combined Symmetry and Secondary Structure Protein")
    st.caption("In this scenario, we initially apply guidance for secondary structure to condition the content accordingly.")
    st.caption("This is followed by incorporating Cyclic symmetry.")
    st.caption("This approach involves adding a secondary structure classifier to conditionally sample an Asymmetric unit (AU) that is beta-rich, followed by symmetrization.")
    
    output='./output/symm_beta.pdb'
    
    CATH=st.sidebar.text_input('CATH:protein domain annotations from <https://www.cathdb.info/>. Annotation examples include 2, 2.40, 2.40.155.','2',key='CATH_beta')
    weight=st.sidebar.number_input('weight : The weighting of the conditioner relative to the backbone model. Defaults is 5,step=1.',value=5,max_value=10,min_value=1,step=1,key='weight')
    max_norm=st.sidebar.number_input(" max_norm: The maximum magnitude of the gradient, above which the magnitude is clipped. Defaults is 20,step=2.",max_value=30,min_value=10,value=20,step=2,key='max_norm')
    beta = conditioners.ProClassConditioner('cath', CATH, weight=weight, max_norm=max_norm)
    symmetry_group=st.sidebar.text_input('symmetry_group:@param ["C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "D_2", "D_3", "D_4", "D_5", "D_6", "D_7", "D_8", "T", "O", "I"]',value="C_3",key='symmetry_group_css')
    knbr=st.sidebar.number_input("knbr:The number of neighbors to consider for each chain in the complex.Default is 2,step=1",min_value=1,max_value=10,step=1,value=2,key='knbr_css')
    c_symmetry = conditioners.SymmetryConditioner(G=symmetry_group, num_chain_neighbors=knbr)
    composed_cond = conditioners.ComposedConditioner([beta, c_symmetry])
    if st.sidebar.button("Run Code with Button",key="substructure"):
        cSSStructureSample(composed_cond,output)
    
    display_by_stmol(output,style,resn)

#  Merging Symmetry and Substructure
def mSSubstructureSample(protein,composedCondtioner,output):
    protein, trajectories = chroma.sample(
        protein_init=protein,
        conditioner=composedCondtioner,
        langevin_factor=4.0,
        langevin_isothermal=True,
        inverse_temperature=8.0,
        sde_func='langevin',
        steps=500,
        full_output=True,
    )
    render(protein,trajectories,output)

def mSSubstructureSampleDemo(style,resn):
    st.sidebar.title("Generate a Merged Symmetry and Substructure Protein")
    st.caption("The goal is to construct symmetric assemblies from a single-chain protein, partially redesigning it to merge three identical AUs into a Cyclic complex.")
    st.caption("We begin by defining the backbones targeted for redesign and then reposition the AU to prevent clashes during symmetrization.")
    st.caption("This is followed by the symmetrization operation itself.")
    
    output='./output/mss_protein.pdb'
    
    pdb_id=st.sidebar.text_input("pdb_id@param ['5SV5', '6QAZ', '3BDI'] {allow-input:true}",'3BDI',key='pdb_id_mss')
    protein = Protein(pdb_id, canonicalize=True, device=device)
    # regenerate residues with X coord < 25 A and y coord < 25 A
    substruct_conditioner = conditioners.SubstructureConditioner(
        protein, backbone_model=chroma.backbone_network, selection="x < 25 and y < 25")

    # C_3 symmetry
    symmetry_group=st.sidebar.text_input('symmetry_group:@param ["C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "D_2", "D_3", "D_4", "D_5", "D_6", "D_7", "D_8", "T", "O", "I"]',value="C_3",key='symmetry_group_mss')
    knbr=st.sidebar.number_input("knbr:The number of neighbors to consider for each chain in the complex.Default is 3,step=1",min_value=1,max_value=10,step=1,value=3,key='knbr_mss')
    c_symmetry = conditioners.SymmetryConditioner(G=symmetry_group, num_chain_neighbors=knbr)

    # Composing
    composed_cond = conditioners.ComposedConditioner([substruct_conditioner, c_symmetry])

    if st.sidebar.button("Run Code with Button",key="substructure"):
        mSSubstructureSample(protein,composed_cond,output)
    
    display_by_stmol(output,style,resn)



# the mapping between the function and use case
demoDict={
    "UnconditionalGenerate":GenerateProteinDemo,
    "ComplexGenerate":complexSampleDemo,
    "SymmetricGenerate":symmetricSampleDemo,
    "ShapeConstrainedGenerate":shapeSampleDemo,
    "FoldConditionGenerate":foldSampleDemo,
    "SecondStructureConditionGenerate":ssSampleDemo,
    "SubstructureConditionGenerate":substructureSampleDemo,
    # "NatureLanguageConditionGenerate":demo.natureLanguageSampleDemo,
    "SymmetryAndSecondaryConditionGenerate":cSSStructureSampleDemo,
    "SymmetryAndSubstructureConditionGenerate":mSSubstructureSampleDemo,
}
# sidebar for demo options
selected_branch = st.sidebar.selectbox("Select the Function to Run:", list(demoDict.keys()))
style=st.sidebar.selectbox("Select Visualization Style:Can be 'stick', 'sphere', 'cross','cartoon'",('stick', 'sphere', 'cross','cartoon'),key='style')
resn=st.sidebar.selectbox("Select the Amino Acid Type to Display",
                          ('*', 'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'),key='resn')

# Exectuion
demoDict[selected_branch](style,resn)



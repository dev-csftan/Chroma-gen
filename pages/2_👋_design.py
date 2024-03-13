import os
import torch
import warnings
from tqdm import tqdm, TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, leave=False)
import pandas as pd

import contextlib



import streamlit as st
from stmol import *

import locale

locale.getpreferredencoding = lambda: "UTF-8"

from chroma import Chroma, Protein, conditioners
from chroma.models import graph_classifier, procap
from chroma.utility.api import register_key
from chroma.utility.chroma import letter_to_point_cloud, plane_split_protein

# api_key = os.environ['API_TOKEN']
api_key='2cdade6d058b4fd1b85fa5badb501312'
register_key(api_key)


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

    # 显示 Protein 的序列
    st.subheader("Protein Sequence:")
    protein_sequence = protein.sequence(format="three-letter-list")
    st.markdown(f"**Protein Sequence:** {protein_sequence}")
    st.write(protein_sequence)
    # 显示 Protein 的结构
    with open(output, "r") as file:
        pdb_content = file.read()

    obj = makeobj(pdb_content,style=style,background='white')

    # 使用 stmol 展示蛋白质结构
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

with contextlib.redirect_stdout(None):
    chroma = Chroma(device=device)

def composeSample(composed_cond,output,**kwargs):

    protein, trajectories = chroma.sample(chain_lengths=[100],
        conditioner=composed_cond,full_output=True,**kwargs)
    
    render(protein, trajectories, output=output)

def composeConditionerSampleDemo(style='',resn=''):
    # output=''
    # display(output,style,resn)
    output='./output/free_protein.pdb'
   
    backboneArgs=selectBackboneArgs()
    sideChainArgs=selectSideChainArgs()
    parameters={**backboneArgs,**sideChainArgs}
    composed_cond=conposeConditioner()

    if st.sidebar.button("Run Code with Button"):
        composeSample(composed_cond,output,**parameters)
    
    display(output,style,resn)

def format_option(option):
    option_explanations = {
        'samples': 'The number of proteins to sample. Default is 1.',
        'steps': 'The number of integration steps for the SDE. Default is 500.',
        'chain_lengths': 'The lengths of the protein chains. Default is [100].',
        'langevin_isothermal': 'Whether to use the isothermal version of the Langevin SDE. Default is False.',
        'integrate_func': 'The name of the integration function to use. Default is “euler_maruyama”.',
        'sde_func': 'The name of the SDE function to use. Defaults to “reverse_sde”.',
        'langevin_factor': 'The factor that controls the strength of the Langevin noise. Default is 2.',
        'inverse_temperature': 'The inverse temperature parameter for the SDE. Default is 10.',
        'protein_init': 'The initial protein state. Defaults to None.',
        'full_output': 'Whether to return the full outputs of the SDE integration, including the protein sample trajectory, the Xhat trajectory (the trajectory of the perceived denoising target) and the Xunc trajectory (the trajectory of the unconditional sample path). Default is False.',
        'initialize_noise': 'Whether to initialize the noise for the SDE integration. Default is True.',
        'tspan': 'The time span for the SDE integration. Default is (1.0, 0.001).',
        'trajectory_length': 'The number of sampled steps in the trajectory output. Maximum is `steps`. Default 200.',
        'design_ban_S': 'List of amino acid single-letter codes to ban, e.g. `["C"]` to ban cysteines.',
        'design_method': 'Specifies which method to use for design. Can be `potts` and `autoregressive`. Default is `potts`.',
        'design_selection': 'Clamp selection for conditioning on a subsequence during sequence sampling. Can be either a selection string or a binary design mask indicating positions to be sampled with shape `(num_batch, num_residues)` or position-specific valid amino acid choices with shape `(num_batch, num_residues, num_alphabet)`.',
        'design_mask_sample': 'Binary design mask indicating which positions can be sampled with shape `(num_batch, num_residues)` or which amino acids can be sampled at which position with shape `(num_batch, num_residues, num_alphabet)`.',
        'design_t': 'Diffusion time for models trained with diffusion augmentation of input structures. Setting `t=0` or `t=None` will condition the model to treat the structure as exact coordinates, while values of `t > 0` will condition the model to treat structures as though they were drawn from noise-augmented ensembles with that noise level. For robust design (default) we recommend `t=0.5`, or for literal design we recommend `t=0.0`. May be a float or a tensor of shape `(num_batch)`.',
        'temperature_S': 'Temperature for sequence sampling. Default 0.01.',
        'temperature_chi': 'Temperature for chi angle sampling. Default 1e-3.',
        'top_p_S': 'Top-p sampling cutoff for autoregressive sampling.',
        'regularization': 'Complexity regularization for sampling.',
        'potts_mcmc_depth': 'Depth of sampling (number of steps per alphabet letter times number of sites) per cycle.',
        'potts_proposal': 'MCMC proposal for Potts sampling. Currently implemented proposals are `dlmc` (default) for Discrete Langevin Monte Carlo [1] or `chromatic` for graph-colored block Gibbs sampling. [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).',
        'potts_symmetry_order': 'Symmetric design. The first `(num_nodes // symmetry_order)` residues in the protein system will be variable, and all consecutively tiled sets of residues will be locked to these during decoding. Internally this is accomplished by summing the parameters Potts model under a symmetry constraint into this reduced sized system and then back imputing at the end. Currently only implemented for Potts models.',
        'SubsequenceConditioner':' Chroma Conditioning module which, given a GraphDesign model and a subset ofresidues for which sequence information is known, can add gradients to samplingthat bias the samples towards increased `log p(sequence | structure)`',
        'ShapeConditioner':'Volumetric potential for optimizing towards arbitrary geometries.',
        'ProCapConditioner':'Natural language conditioning for protein backbones.This conditioner uses an underlying `ProteinCaption` model to determine thelikelihood of a noised structure corresponding to a given caption. Captionscan be specified as corresopnding to a particular chain of the structure, or to the entire complex. The encoded structures and captions are passed to themodel together, and the output loss that adjusts the energy is the masked cross-entropy over the caption tokens.',
        'ProClassConditioner':'A Chroma Conditioning module which can specify chain level annotations for fold,function, and organism. The current labels that can be conditioned on are:',
        'SubstructureConditioner':'A Chroma Conditioning module which can specifiy a subset of residues for which to condition on absolute atomic coordinates, see supplementary section M for more details.',
        'SymmetryConditioner':' A symmetry conditioner applies a set of symmetry operations to a protein structure and enforces constraints on the resulting conformations. It can be used to model symmetric complexes or assemblies of proteins.',
        'verbose':' bool = False',
    }
    return f"{option} - {option_explanations.get(option, 'No explanation available')}"

def selectBackboneArgs():
     # backbone args 
    """
    # Backbone Args
        samples: int = 1,
        steps: int = 500,
        chain_lengths: List[int] = [100],
        tspan: List[float] = (1.0, 0.001),
        protein_init: Protein = None,
        conditioner: Optional[nn.Module] = None,
        langevin_factor: float = 2,
        langevin_isothermal: bool = False,
        inverse_temperature: float = 10,
        initialize_noise: bool = True,
        integrate_func: Literal["euler_maruyama", "heun"] = "euler_maruyama",
        sde_func: Literal["langevin", "reverse_sde", "ode"] = "reverse_sde",
        trajectory_length: int = 200,
        full_output: bool = False,
    """
    options=st.sidebar.multiselect('Choose backbone parameters for sampling',
        ['samples','steps', 'chain_lengths','tspan', 'langevin_factor',
         'langevin_isothermal','inverse_temperature','trajectory_length','protein_init',
         'initialize_noise','integrate_func','sde_func'],
        [],format_func=format_option)
    container=st.sidebar.container(border=True)
    
    parameters={}
    if 'steps' in options:
        parameters['steps']=container.number_input("steps:The number of integration steps for the SDE. Default is 500.",min_value=150,max_value=500,step=50,value=200,key='steps_sample')
    if 'samples' in options:
        parameters['samples']=container.number_input("samples:The number of proteins to sample. Default is 1",min_value=1,max_value=5,value=1,step=1,key='samples')
    if 'chain_lengths' in options:
        parameters['chain_lengths'] = container.number_input("chain_lengths:The lengths of the protein chains. Default is [100].",min_value=50,max_value=350, step=50,value=100,key='chain_lengths')
    
    if 'tspan' in options:
        parameters['tspan'] = container.slider('tspan:Select time span', min_value=0.001, max_value=1.0, value=(1.0, 0.001),step=0.001)

    if 'langevin_factor' in options:
        parameters['langevin_factor'] = container.number_input('langevin_factor:The factor that controls the strength of the Langevin noise. Default is 2.',min_value=1.0, max_value=5.0, value=2.0, step=1,
                                                               key='langevin_factor')

    if 'langevin_isothermal' in options:
        parameters['langevin_isothermal'] = container.checkbox("langevin_isothermal:Whether to use the isothermal version of the Langevin SDE. Default is False",value=False)

    if 'inverse_temperature' in options:
        parameters['inverse_temperature'] = container.number_input('inverse_temperature:The inverse temperature parameter for the SDE. Default is 10.',min_value=0, max_value=40, value=10, step=2,
                                                                   key='inverse_temperature')

    if 'initialize_noise' in options:
        parameters['initialize_noise'] = container.checkbox('initialize_noise:Whether to initialize the noise for the SDE integration. Default is True.',value=True)

    if 'integrate_func' in options:
        parameters['integrate_func'] = container.selectbox('integrate_func:The name of the integration function to use. Default is “euler_maruyama”', ['euler_maruyama', 'heun'],
                                                           index=0)

    if 'sde_func' in options:
        parameters['sde_func'] = container.selectbox('sde_func:Select SDE Function', ['langevin', 'reverse_sde', 'ode'], index=0)

    if 'trajectory_length' in options:
        parameters['trajectory_length'] = container.number_input('trajectory_length:The number of sampled steps in the trajectory output. Maximum is `steps`. Default 200.',min_value=50, max_value=300, value=200, step=50,
                                                                key='trajectory_length')

    if 'protein_init' in options:
        pdb_id=st.sidebar.text_input("protein_init: Select pdb_id@param ['5SV5', '6QAZ', '3BDI'] {allow-input:true}",'3BDI',key='pdb_id')
        protein = Protein(pdb_id, canonicalize=True, device=device)
        parameters['protein_init'] = protein

    
    return parameters
    
def selectSideChainArgs():
    """
     # Sidechain Args
        design_ban_S: Optional[List[str]] = None,
        design_method: Literal["potts", "autoregressive"] = "potts",
        design_selection: Optional[Union[str, torch.Tensor]] = None,
        design_t: Optional[float] = 0.5,
        temperature_S: float = 0.01,
        temperature_chi: float = 1e-3,
        top_p_S: Optional[float] = None,
        regularization: Optional[str] = "LCP",
        potts_mcmc_depth: int = 500,
        potts_proposal: Literal["dlmc", "chromatic"] = "dlmc",
        potts_symmetry_order: int = None,
        verbose: bool = False,
    """
    options = st.sidebar.multiselect('Choose side_chain parameters for sampling',
        ['design_ban_S', 'design_method', 'design_selection', 'design_t', 'temperature_S', 'temperature_chi',
         'top_p_S', 'regularization', 'potts_mcmc_depth', 'potts_proposal', 'potts_symmetry_order', 'verbose'],
        [], format_func=format_option, key='sideChainArgs')
    container = st.sidebar.container(border=True)
    
    parameters = {}
    
    if 'design_ban_S' in options:
        # Placeholder value, modify according to your requirements
        parameters['design_ban_S'] = container.multiselect('design_ban_S:Select banned residues for design', ['Res1', 'Res2'], default=[])

    if 'design_method' in options:
        # Placeholder value, modify according to your requirements
        parameters['design_method'] = container.radio('design_method:Select design method', ['potts', 'autoregressive'], index=0)

    if 'design_selection' in options:
        # Placeholder value, modify according to your requirements
        parameters['design_selection'] = container.text_input('design_selection:Enter design selection', key='design_selection')

    if 'design_t' in options:
        parameters['design_t'] = container.number_input('design_t:Enter design temperature', min_value=0.0, value=0.5, key='design_t')

    if 'temperature_S' in options:
        parameters['temperature_S'] = container.number_input('temperature_S:Enter S temperature', value=0.01, key='temperature_S')

    if 'temperature_chi' in options:
        parameters['temperature_chi'] = container.number_input('temperature_chi:Enter chi temperature', value=1e-3, key='temperature_chi')

    if 'top_p_S' in options:
        # Placeholder value, modify according to your requirements
        parameters['top_p_S'] = container.number_input('top_p_S:Enter top p for S', key='top_p_S')

    if 'regularization' in options:
        # Placeholder value, modify according to your requirements
        parameters['regularization'] = container.selectbox('regularization:Select regularization method', ['LCP', 'Other'], index=0)

    if 'potts_mcmc_depth' in options:
        parameters['potts_mcmc_depth'] = container.number_input('potts_mcmc_depth:Enter MCMC depth for Potts', value=500, key='potts_mcmc_depth')

    if 'potts_proposal' in options:
        # Placeholder value, modify according to your requirements
        parameters['potts_proposal'] = container.selectbox('potts_proposal:Select Potts proposal', ['dlmc', 'chromatic'], index=0)

    if 'potts_symmetry_order' in options:
        parameters['potts_symmetry_order'] = container.number_input('potts_symmetry_order:Enter symmetry order for Potts', key='potts_symmetry_order')

    if 'verbose' in options:
        parameters['verbose'] = container.checkbox('verbose:Enable Verbose', value=False)

    return parameters



def conposeConditioner():
    options = st.sidebar.multiselect(
        'Choose conditioners for sampling',
        ['ProClassConditioner', 'SymmetryConditioner', 'SubsequenceConditioner', 
         'ProCapConditioner','SubstructureConditioner'],
        [],format_func=format_option)
    conditioners_list=[]

        # 判断每个选项是否被选择
    if 'ProClassConditioner' in options:
        container=st.sidebar.container(border=True)
        container.write('----ProClassConditioner is selected!------')
        CATH=container.text_input('CATH:protein domain annotations from <https://www.cathdb.info/>. Annotation examples include 2, 2.40, 2.40.155.','3.40.50',key='CATH')
        length=container.number_input('chain_length:The lengths of the protein chains.Default is 130,step=10.',min_value=50,max_value=250,step=10,value=130,key='length_fold')

        proclass_model = graph_classifier.load_model("named:public", device=device)
        conditioner = conditioners.ProClassConditioner("cath", CATH, model=proclass_model,device=device)
        container.write('----ProClassConditioner is selected!------')
        conditioners_list.append(conditioner)

    if 'SymmetryConditioner' in options:
        container=st.sidebar.container(border=True)
        container.write('----SymmetryConditioner is selected!------')
        symmetry_group=container.text_input('symmetry_group:@param ["C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "D_2", "D_3", "D_4", "D_5", "D_6", "D_7", "D_8", "T", "O", "I"]',value="C_3",key='symmetry_group_mss')
        knbr=container.number_input("knbr:The number of neighbors to consider for each chain in the complex.Default is 3,step=1",min_value=1,max_value=10,step=1,value=3,key='knbr_mss')
        c_symmetry = conditioners.SymmetryConditioner(G=symmetry_group, num_chain_neighbors=knbr)
        container.write('----SymmetryConditioner is selected!------')
        conditioners_list.append(c_symmetry)
    
    if 'SubsequenceConditioner' in options:
        pdb_id=st.sidebar.text_input("pdb_id@param ['5SV5', '6QAZ', '3BDI'] {allow-input:true}",'3BDI',key='pdb_id_mss')
        protein = Protein(pdb_id, canonicalize=True, device=device)
        # regenerate residues with X coord < 25 A and y coord < 25 A
        substruct_conditioner = conditioners.SubstructureConditioner(
            protein, backbone_model=chroma.backbone_network, selection="x < 25 and y < 25")
        conditioners_list.append(substruct_conditioner)
    
    if 'ProCapConditioner' in options:
        CAPTION=st.sidebar.text_input('a caption:natural language prompts.',value='Crystal structure of SH2 domain',key='caption')
        torch.manual_seed(0)
        conditioner = conditioners.ProCapConditioner(CAPTION, -1)
        conditioners_list.append(conditioner)

    if 'SubstructureConditioner' in options:
        pdb_id=st.sidebar.text_input("pdb_id@param ['5SV5', '6QAZ', '3BDI'] {allow-input:true}",'3BDI',key='pdb_id_mss')
        protein = Protein(pdb_id, canonicalize=True, device=device)
        # regenerate residues with X coord < 25 A and y coord < 25 A
        substruct_conditioner = conditioners.SubstructureConditioner(
            protein, backbone_model=chroma.backbone_network, selection="x < 25 and y < 25")
        
    # if 'ScrewConditioner' in options:

    # if 'InflateConditioner' in options:
    # if 'RgConditioner' in options:

    composed_cond = conditioners.ComposedConditioner(conditioners_list)
    return composed_cond



style=st.sidebar.selectbox("Select Visualization Style:Can be 'stick', 'sphere', 'cross','cartoon'",('stick', 'sphere', 'cross','cartoon'),key='style')
resn=st.sidebar.selectbox("Select the Amino Acid Type to Display",
                          ('*', 'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'),key='resn')
composeConditionerSampleDemo(style,resn)
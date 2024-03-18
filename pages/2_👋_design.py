from utils import *

st.set_page_config(
    page_title="GenAI for Protein Design",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Protein Design Driven by Diffusion Model")


def composeSample(composed_cond,output,**kwargs):

    protein, trajectories = chroma.sample(chain_lengths=[100],
        conditioner=composed_cond,full_output=True,**kwargs)
    
    render(protein, trajectories, output=output)

def composeConditionerSampleDemo(style,resn):
    st.caption("Here, Our objective is to facilitate scientists in the unrestricted design of protein.")

    output='./output/free_protein.pdb'

    main_para_container=st.sidebar.container(border=True)
    main_para_container.title('Sampling parameters')
    main_para_container.caption(""" Performs Backbone Sampling and Sequence Design and returns a Protein or list
                                    of Proteins. Optionally this method can return additional arguments to show
                                    details of the sampling procedure.""")
    backboneArgs=selectBackboneArgs(main_para_container)
    sideChainArgs=selectSideChainArgs(main_para_container)
    parameters={**backboneArgs,**sideChainArgs}

    main_cond_container=st.sidebar.container(border=True)
    main_cond_container.title('Composed Conditioners')
    composed_cond=conposeConditioner(main_cond_container)

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
        'design_method': 'Specifies which method to use for design the side_chain. Can be `potts` and `autoregressive`. Default is `potts`.',
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
        'SubsequenceConditioner':' Conditioning module which, given a GraphDesign model and a subset ofresidues for which sequence information is known, can add gradients to samplingthat bias the samples towards increased `log p(sequence | structure)`',
        'ShapeConditioner':'Volumetric potential for optimizing towards arbitrary geometries.',
        'ProCapConditioner':'Natural language conditioning for protein backbones.This conditioner uses an underlying `ProteinCaption` model to determine thelikelihood of a noised structure corresponding to a given caption. Captionscan be specified as corresopnding to a particular chain of the structure, or to the entire complex. The encoded structures and captions are passed to themodel together, and the output loss that adjusts the energy is the masked cross-entropy over the caption tokens.',
        'ProClassConditioner':'A Conditioning module which can specify chain level annotations for fold,function, and organism. The current labels that can be conditioned on are:',
        'SubstructureConditioner':'A Conditioning module which can specifiy a subset of residues for which to condition on absolute atomic coordinates, see supplementary section M for more details.',
        'SymmetryConditioner':' A symmetry conditioner applies a set of symmetry operations to a protein structure and enforces constraints on the resulting conformations. It can be used to model symmetric complexes or assemblies of proteins.',
        'verbose':' bool = False',
    }
    return f"{option} - {option_explanations.get(option, 'No explanation available')}"


def selectBackboneArgs(main_para_container):
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
    container=main_para_container.container(border=True)
    container.title('Backbone Args')
    options=container.multiselect(label='Set the backbone arguments for sampling',
        options=['samples','steps', 'chain_lengths','tspan', 'langevin_factor',
         'langevin_isothermal','inverse_temperature','trajectory_length','protein_init',
         'initialize_noise','integrate_func','sde_func'],
        default=[],format_func=format_option)
    

    parameters={}
    if 'steps' in options:
        parameters['steps']=container.number_input("steps:The number of integration steps for the SDE. Default is 500.",min_value=150,max_value=500,step=50,value=200,key='steps_sample')
    if 'samples' in options:
        parameters['samples']=container.number_input("samples:The number of proteins to sample. Default is 1",min_value=1,max_value=5,value=1,step=1,key='samples')
    if 'chain_lengths' in options:
        parameters['chain_lengths'] = container.number_input("chain_lengths:The lengths of the protein chains. Default is [100].",min_value=50,max_value=350, step=50,value=100,key='chain_lengths')
    
    if 'tspan' in options:
        subcontainer=container.container(border=True)
        subcontainer.caption("""tspan (Tuple[float, float], optional): Time interval over which to appl
                                y reconstruction guidance, can be helpful to turn off at times close to
                                zero. tspan[0] should be < tspan[1].""")
        start=subcontainer.number_input('tspan[0]',value=1e-1,key='start')
        end=subcontainer.number_input('tspan[1]',value=1,key='end')
        tspan=tuple([start,end])
        parameters['tspan']=tspan

    if 'langevin_factor' in options:
        parameters['langevin_factor'] = container.number_input('langevin_factor:The factor that controls the strength of the Langevin noise. Default is 2.',min_value=1, max_value=10, value=2, step=1,
                                                               key='langevin_factor')

    if 'langevin_isothermal' in options:
        parameters['langevin_isothermal'] = container.selectbox("langevin_isothermal:Whether to use the isothermal version of the Langevin SDE. Default is False",[True,False],index=1,key='langevin_isothermal')

    if 'inverse_temperature' in options:
        parameters['inverse_temperature'] = container.number_input('inverse_temperature:The inverse temperature parameter for the SDE. Default is 10.',min_value=0, max_value=40, value=10, step=2,
                                                                   key='inverse_temperature')

    if 'initialize_noise' in options:
        parameters['initialize_noise'] = container.selectbox('initialize_noise:Whether to initialize the noise for the SDE integration. Default is True.',[True,False],index=0,key='initialize_noise')

    if 'integrate_func' in options:
        parameters['integrate_func'] = container.selectbox('integrate_func:The name of the integration function to use. Default is “euler_maruyama”', ['euler_maruyama', 'heun'],
                                                           index=0)

    if 'sde_func' in options:
        parameters['sde_func'] = container.selectbox('sde_func:Select SDE Function', ['langevin', 'reverse_sde', 'ode'], index=0)

    if 'trajectory_length' in options:
        parameters['trajectory_length'] = container.number_input('trajectory_length:The number of sampled steps in the trajectory output. Maximum is `steps`. Default 200.',min_value=50, max_value=300, value=200, step=50,
                                                                key='trajectory_length')

    if 'protein_init' in options:
        pdb_id=container.selectbox("protein_init: Select pdb_id@param ['5SV5', '6QAZ', '3BDI'] {allow-input:true}",['5SV5', '6QAZ', '3BDI'],index=2,key='pdb_id')
        protein = Protein(pdb_id, canonicalize=True, device=device)
        parameters['protein_init'] = protein

    
    return parameters
    
def selectSideChainArgs(main_para_container):
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
        verbose: bool = False,
    """
    container = main_para_container.container(border=True)
    container.title('Sidechain Args')
    options = container.multiselect('Set the side_chain arguments for sampling',
        ['design_ban_S', 'design_method', 'design_selection', 'design_t', 'temperature_S', 'temperature_chi',
         'top_p_S', 'regularization', 'potts_mcmc_depth', 'potts_proposal',  'verbose'],
        [], format_func=format_option, key='sideChainArgs')
    
    
    parameters = {}
    
    if 'design_ban_S' in options:
        parameters['design_ban_S'] = container.multiselect('design_ban_S:Select banned residues for design', ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'], default=[])

    if 'design_method' in options:
        parameters['design_method'] = container.selectbox('design_method:Select design method', ['potts', 'autoregressive'], index=0,key='design_method')

    if 'design_selection' in options:
        parameters['design_selection'] = container.text_input("""design_selection(str or torch.Tensor, optional): Clamp selection for
                                                                conditioning on a subsequence during sequence sampling. Can be
                                                                either a selection string or a binary design mask indicating
                                                                positions to be sampled with shape `(num_batch, num_residues)` or
                                                                position-specific valid amino acid choices with shape
                                                                `(num_batch, num_residues, num_alphabet)`. """,
                                                                value="resid 20-50 around 5.0", key='design_selection')


    if 'design_t' in options:
        parameters['design_t'] = container.number_input("""design_t(float or torch.Tensor, optional): Diffusion time for models
                                                        trained with diffusion augmentation of input structures. Setting `t=0`
                                                        or `t=None` will condition the model to treat the structure as
                                                        exact coordinates, while values of `t > 0` will condition
                                                        the model to treat structures as though they were drawn from
                                                        noise-augmented ensembles with that noise level. For robust design
                                                        (default) we recommend `t=0.5`, or for literal design we recommend
                                                        `t=0.0`. May be a float or a tensor of shape `(num_batch)`.""", 
                                                        min_value=0.0, value=0.5, key='design_t')

    if 'temperature_S' in options:
        parameters['temperature_S'] = container.number_input('temperature_S:Temperature for sequence sampling.Default 0.01.', min_value=0.0,value=0.01, key='temperature_S')

    if 'temperature_chi' in options:
        parameters['temperature_chi'] = container.number_input('temperature_chi:Temperature for chi angle sampling.Default 1e-3.', value=1e-3, key='temperature_chi')

    if 'top_p_S' in options:
        parameters['top_p_S'] = container.number_input('top_p_S(float, optional)Top-p sampling cutoff for autoregressiv sampling.', value=0.8,key='top_p_S')

    if 'regularization' in options:
        parameters['regularization'] = container.selectbox('regularization(str, optional): Complexity regularization for sampling.', ['LCP'], index=0)

    if 'potts_mcmc_depth' in options:
        parameters['potts_mcmc_depth'] = container.number_input('potts_mcmc_depth(int, optional)Enter MCMC depth for Potts per cycle', min_value=50,value=500, step=50,key='potts_mcmc_depth')

    if 'potts_proposal' in options:
        parameters['potts_proposal'] = container.selectbox("""potts_proposal:MCMC proposal for Potts sampling. Currently implemented
                    proposals are `dlmc` (default) for Discrete Langevin Monte Carlo or
                    `chromatic` for graph-colored block Gibbs sampling.""", 
                    ['dlmc', 'chromatic'], index=0)


    if 'verbose' in options:
        parameters['verbose'] = container.selectbox('verbose:If True print verbose output during sampling.', [True,False],index=1,key='verbose')

    return parameters

def conposeConditioner(main_cond_container):
    options = main_cond_container.multiselect(
                                    'Choose conditioners for sampling',
                                    ['ProClassConditioner', 
                                    'SymmetryConditioner',
                                    'SubsequenceConditioner', 
                                    # 'ProCapConditioner',
                                    'SubstructureConditioner',
                                    'ScrewConditioner',
                                    'InflateConditioner',
                                    'RgConditioner',
                                    'ShapeConditioner'],
                                    [])
    
    conditioners_list=[]

    if 'ProClassConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('ProClassConditioner')
        container.caption('A Conditioning module which can specify chain level annotations for fold,function, and organism. ')
        container.caption("""Note:This conditioner is a research preview. Conditioning with it can be inconsistent
        and depends on the relative prevalence of a given label in the dataset.
        With repeated tries it will produce successful results for more abundant labels.
        Please see the supplement to the paper for details. This is currently not
        recommended for production use. The most reproducible labels are C level
        annotations in cath, (e.g. `1`,`2`,`3`).""")


        label =container.selectbox('label:The annotation to condition on in the set [cath, secondary_structure].',
                                   ['cath','secondary_structure'],index=0,key='label')
        proclass_model = graph_classifier.load_model("named:public", device=device)
        
        if label is 'cath':
            CATH=container.selectbox('CATH:protein domain annotations from <https://www.cathdb.info/>. Annotation examples include 2, 2.40, 2.40.155.',['2','2.40','2.40.155'],index=2,key='CATH')
            weight=container.number_input('weight : The weighting of the conditioner relative to the backbone model. Defaults is 5,step=1.',value=5,max_value=10,min_value=1,step=1,key='weight')
            max_norm=container.number_input(" max_norm: The maximum magnitude of the gradient, above which the magnitude is clipped. Defaults is 20,step=2.",max_value=30,min_value=10,value=20,step=2,key='max_norm')
            conditioner = conditioners.ProClassConditioner(label, CATH, model=proclass_model,device=device,weight=weight,max_norm=max_norm)
        else :
            container.caption('Smoke test for secondary structure conditioning')
            SECONDARY_STRUCTURE=container.text_area('secondary srtucture',
                                                     value='CCEEEEEEEETTTTECTTTTTTTTCCCHHHHHHHHHHHHCCCTTTTEEEEEECHHHHHHCTGGTTTTTTTEEEEETTTTTTTTTTTCEEECTHHHHHHHHHCHGHGGHCCEEEEEECHHHHHHHHHCTCEEEEEEEEETTCCCTTEECCCCTGGGTEEETETTTTTCCEEEETTEEEEEEEEEEEC',
                                                     key='ss')
            
            conditioner = conditioners.ProClassConditioner(label,SECONDARY_STRUCTURE, model=proclass_model,device=device)
        conditioners_list.append(conditioner)


    if 'SymmetryConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('SymmetryConditioner')
        container.caption('A class that implements a symmetry conditioner for a protein structure.')
        container.caption("""A symmetry conditioner applies a set of symmetry operations to a protein structure
                            and enforces constraints on the resulting conformations. It can be used to model
                            symmetric complexes or assemblies of proteins.""")
        

        symmetry_group=container.selectbox('symmetry_group:str representing the symmetry operations as rotation matrices.',
                                            ["C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "D_2", "D_3", "D_4", "D_5", "D_6", "D_7", "D_8", "T", "O", "I"],index=1,key='symmetry_group_mss')
        
        knbr=container.number_input("knbr(int): The number of neighbors to consider for each chain in the complex.",min_value=1,max_value=10,step=1,value=3,key='knbr_mss')

        freeze_com=container.selectbox('freeze_com(bool):Whether to freeze the center of mass of the complex during optimization.',[True,False],index=1,key='freeze_com')
        grad_com_surgery=container.selectbox('grad_com_surgery(bool): Whether to apply gradient surgery to remove the center of mass component from the gradient.',[True,False],index=1,key='grad_com_surgery')
        interface_restraint=container.selectbox('interface_restraint(bool): Whether to apply a flat-bottom potential to restrain the distance between neighboring chains in the complex.',[True,False],index=1,key='interface_restraint')
        restraint_grad=container.selectbox('restraint_grad(bool): Whether to include the restraint gradient in the totalgradient.',[True,False],index=1,key='restraint_grad')
        enable_rigid_drift=container.selectbox('enable_rigid_drift(bool): Whether to enable rigid body drift correction for the complex.',[True,False],index=0,key='enable_rigid_drift')
        canonicalize=container.selectbox('canonicalize(bool): Whether to canonicalize the chain order and orientation of the complex.',[True,False],index=0,key='canonicalize')

        c_symmetry = conditioners.SymmetryConditioner(G=symmetry_group, 
                                                      num_chain_neighbors=knbr,
                                                      freeze_com=freeze_com,
                                                      grad_com_surgery=grad_com_surgery,
                                                      interface_restraint=interface_restraint,
                                                      restraint_grad=restraint_grad,
                                                      enable_rigid_drift=enable_rigid_drift,
                                                      canonicalize=canonicalize)
        
        conditioners_list.append(c_symmetry)
    
    if 'SubsequenceConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('SubsequnceConditoner')
        container.caption("""A Conditioning module which, given a GraphDesign model and a subset of
                            residues for which sequence information is known, can add gradients to sampling
                            that bias the samples towards increased `log p(sequence | structure)`""")
        
        pdb_id=container.selectbox("pdb_id(str): The PDBID of the protein to fetch.", ['5SV5', '6QAZ', '3BDI'],index=0,key='pdb_id_subseq')
        protein = Protein(pdb_id, canonicalize=True, device=device)

        selection=container.selectbox('selection(str):A selection string to specify which residues should be included in the mask.',['hyd','all','none'],index=1,key='selection')
        weight=container.number_input('weight(float, optional): Overall weight to which the gradient is scaled.',value=1.0,step=0.1,min_value=0.0)
        renormalize_grad=container.selectbox('renormalize_grad(bool, optional): Whether to renormalize gradient to have overall variance `weight`.',[True,False],index=1,key='renormalize_grad')

        subConditioner=conditioners.SubsequenceConditioner(protein=protein,
                                                           design_model=GraphDesign(predict_S_marginals=True, predict_S_potts=True),
                                                           selection=selection,
                                                           weight=weight,
                                                           renormalize_grad=renormalize_grad)
        conditioners_list.append(subConditioner)
    
    if 'ProCapConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('ProCapConditioner')
        container.caption("Natural language conditioning for protein backbones.")
        container.caption("""This conditioner uses an underlying `ProteinCaption` model to determine the
                            likelihood of a noised structure corresponding to a given caption. Captions
                            can be specified as corresopnding to a particular chain of the structure, or
                            to the entire complex. The encoded structures and captions are passed to the
                            model together, and the output loss that adjusts the energy is the masked
                            cross-entropy over the caption tokens.""")
        
        CAPTION=container.text_input("""CAPTION:Caption for the conditioner. Currently, a separate
                                        conditioner should be constructed for each desired caption, even
                                        with a single `ProteinCaption` model.""",
                                        value='Crystal structure of SH2 domain',key='caption')
        chain_id=container.number_input("""chain_id(int): The 1-indexed chain to which the caption corresponds, or
                                            -1 for captions corresponding to the entire structure. The provided
                                            checkpoints are trained with UniProt captions for chain_id > 0 and
                                            PDB caption for chain_id = -1. Regardless of whether the caption is
                                            specific to one chain, the conditioner acts on the entire structure.""",
                                             value=-1,key='chain_id')
        weight=container.number_input('weight(float): Overall factor by which the caption gradient is scaled.',value=10.0,key='weight')

        use_sequence=container.selectbox('use_sequence(bool): Whether to use input sequence, default False.',[True,False],index=1,key='use_sequence')

        torch.manual_seed(0)

        conditioner = conditioners.ProCapConditioner(caption=CAPTION,
                                                     chain_id=chain_id,
                                                     weight=weight,
                                                     use_sequence=use_sequence,
                                                     device=device)
        
        conditioners_list.append(conditioner)


    if 'SubstructureConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('SubstructureConditioner')
        container.caption("""A Conditioning module which can specifiy a subset of residues for which to
                            condition on absolute atomic coordinates, see supplementary section M for more
                            details.""")

        pdb_id=container.selectbox("pdb_id(str): The PDBID of the protein to fetch.", ['5SV5', '6QAZ', '3BDI'],index=0,key='pdb_id_substr')
        protein = Protein(pdb_id, canonicalize=True, device=device)

        selection=container.text_input('selection(str):A selection string to specify which residues should be included in the mask.',
                                       value="x < 25 and y < 25",key='selection_substr')
        
        rg=container.selectbox("""rg(bool, optional): Whether or not to add reconstruction guidance gradients,
                                see supplementary section M for a discussion. This can reduce incidence of
                                clashes / bond violations / discontinuities at the cost of inference time
                                and some stability.""",
                               [True,False],index=1,key='rg')
        weight=container.number_input('weight(float, optional): Overall weight of the reconstruction guidance term (untransformed).',
                                      value=1.0,min_value=0.0,key='weight')

        subcontainer=container.container(border=True)
        subcontainer.caption("""tspan (Tuple[float, float], optional): Time interval over which to appl
                                y reconstruction guidance, can be helpful to turn off at times close to
                                zero. tspan[0] should be < tspan[1].""")
        start=subcontainer.number_input('tspan[0]',value=1e-1,key='start')
        end=subcontainer.number_input('tspan[1]',value=1,key='end')
        tspan=tuple([start,end])

        weight_max=container.number_input(' weight_max:Final rg gradient is rescaled to have `scale`variance, where `scale` is clamped to have a maximum value of `max_weight`.',
                                          value=3.0,step=0.1,key='weight_max')
        
        gamma=container.selectbox(""" gamma (Optional[float]): Gamma inflates the translational degree of freedom
                                    of the underlying conditional multivariate normal, making it easier for
                                    model to move the center of mass of the infilled samples.
                                    Setting to [0.01, 0.1, 1.0] is a a plausible place to start to increase
                                    sample Rg.""",
                                    [0.01, 0.1, 1.0],key='gamma')
        center_init=container.selectbox('center_init (Optional[bool]): Whether to center the input structural data',
                                        [True,False],index=0,key='center_init')
        
        
        # regenerate residues with X coord < 25 A and y coord < 25 A
        substruct_conditioner = conditioners.SubstructureConditioner(
                                                                    protein=protein, 
                                                                    backbone_model=chroma.backbone_network,
                                                                    selection=selection,
                                                                    rg=rg,
                                                                    weight=weight,
                                                                    tspan=tspan,
                                                                    weight_max=weight_max,
                                                                    gamma=gamma,
                                                                    center_init=center_init)
        conditioners_list.append(substruct_conditioner)
        
    if 'ScrewConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('ScrewConditioner')
        container.caption('A class that implements a screw conditioner for a protein structure')
        container.caption("""A screw conditioner applies a screw transformation to a protein structure
                            and repeats it for a given number of times. It can be used to model
                            helical or cyclic symmetry of proteins.""")
        
        theta=container.number_input('theta (float): The angle of rotation about the z-axis in radians.',value=np.pi / 4,step=np.pi/4,key='theta')
        tz=container.number_input('tz (float): The translation along the z-axis.',value=5.0,key='tz')
        M=container.number_input('M (int): The number of repetitions of the screw transformation.',value=10,key='M')
        
        screwConditioner=conditioners.ScrewConditioner(theta=theta,tz=tz,M=M)

        conditioners_list.append(screwConditioner)

    if 'InflateConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('Inflate conditioner')
        container.caption("""This class inherits from the Conditioner class and defines a specific conditioner
                                that inflates shift the COM of X based on a vector v and a scalar.""")
        

        tensor_input = container.text_area("v (torch.Tensor): Vector to add to X with shape `(num_residues, 4, 3)`.\nPlease enter values separated by commas",value="1,4,3")
        parsed_list = tensor_input.split(',')
        parsed_list = [float(num) for num in parsed_list]
        v=torch.tensor(parsed_list)

        scale=container.number_input('scale (float): Scale factor for v.',key='scale')
        inflatConditioner=conditioners.InflateConditioner(v=v,scale=scale)

        conditioners_list.append(inflatConditioner)
        
    if 'RgConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('RgConditioner')
        container.caption('Conditioners that penalized backbones for having Rg deviated from the expected Rg Scaling.')
        container.caption(""""The penalty function takes the form of a flat bottom potential
                            penalty = || ReLU( || Rg(X, C) - Rg_ceiling_scale * expected_Rg(C) || ) ||^2""")

        scale=container.number_input('scale (float): Scale factor for the penalty',value=1,key='scale')
        Rg_ceiling_scale=container.number_input('Rg_ceiling_scale (float): the flat bottom potentialy width, needs to be larger than 1.',value=1.5,key='Rg_ceiling_scale')
        complex_rg=container.selectbox("""complex_rg(bool): whether compute expected Rg based on the complex Rg scaling.
                                        If True, expected Rg will be computed by treating the entire complex as if
                                        it is a single cahin. If False, expected Rg will be computed for individual
                                        chains""",
                                       [True,False],index=1)
        rgContainer=conditioners.RgConditioner(scale=scale,
                                               Rg_ceiling_scale=Rg_ceiling_scale,
                                               complex_rg=complex_rg)
        conditioners_list.append(rgContainer)


    if 'ShapeConditioner' in options:
        container=main_cond_container.container(border=True)
        container.title('ShapeCondtioner')
        container.caption('Volumetric potential for optimizing towards arbitrary geometries.')
        
        character=container.text_input('character:a desired character for the shape of protein.','G',key='character')
        X_target=letter_to_point_cloud(character)

        noise_schedule=chroma.backbone_network.noise_schedule

        autoscale=container.selectbox("""autoscale (bool): If True, automatically rescale target point cloud coordinates
                                            such that they are approximately volume-scaled to a target protein size.
                                            Volume is roughly estimated by converting the point cloud to a sphere cloud
                                            with radii large enough to overlap with near neighbors and double counting
                                            corrections via inclusion-exclusion.""",
                                            [True,False],index=1,key='autoscale')
        autoscale_num_residues=container.number_input('autoscale_num_residues (int): Target protein size for auto-scaling.',
                                                      value=500,step=100,key='autoscale_num_residus')
        autoscale_target_ratio=container.number_input('autoscale_target_ratio (float): Scale factor for adjusting the target protein volume.',
                                                      value=0.4,step=0.1,min_value=0.0,key='targer_ratio')
        scale_invariant=container.selectbox("""scale_invariant (bool): If True, compute the loss in a size invariant manner
                                                by dynamically renormalizing the point clouds to match Radii of gyration.
                                                This approach can be more unstable to integrate and require more careful tuning.""",
                                                [True,False],index=1,key='scale_invariant')
        shape_loss_weight=container.number_input('shape_loss_weight (float): Scale factor for the overall restraint.',
                                                 value=20.0,key='shape_loss_weight')
        shape_loss_cutoff=container.number_input('shape_loss_cutoff (float): Minimal distance deviation that is penalized in the loss, e.g. to treat as a flat-bottom restraint below the cutoff.',
                                                 value=0.0,min_value=0.0,key='shape_loss_cutoff')
        sinkhorn_scale=container.number_input('sinkhorn_scale (float): Entropy regularization scaling parameter for Optimal Transport calculations.',
                                              value=1.0,key='sinkhron_scale')
        sinkhorn_iterations_gw=container.number_input('sinkhorn_iterations_gw (int): Number of Sinkhorn iterations for Gromov-Wasserstein Optimal Transport calculations.',
                                                      value=30,step=10,key='sinkhorn_iterations_gw')
        sinkhorn_scale_gw=container.number_input('sinkhorn_scale_gw (float): Entropy regularization scaling parameter for Gromov-Wasserstein Optimal Transport calculations.',
                                                 value=200.0,step=50.0,key='sinkhorn_scale_gw')
        gw_layout=container.selectbox("""gw_layout (bool): If True, use Gromov-Wasserstein Optimal Transport to compute
                                        a point cloud correspondence assuming ideal protein distance scaling.""",
                                        [True,False],index=0,key='gw_layout')
        gw_layout_coefficient=container.number_input("""gw_layout_coefficient (float): Scale factor with which to combine average
                                                        inter-point cloud distances according to OT (Wasserstein) versus
                                                        Gromov-Wasserstein couplings.""",value=0.4,step=0.1,key='gw_layout_coefficient')
        
        
        conditioner=conditioners.ShapeConditioner(X_target=X_target,
                                                  noise_schedule=noise_schedule,
                                                  autoscale=autoscale,autoscale_num_residues=autoscale_num_residues,
                                                  scale_invariant=scale_invariant,
                                                  shape_loss_weight=shape_loss_weight,
                                                  shape_loss_cutoff=shape_loss_cutoff,
                                                  autoscale_target_ratio=autoscale_target_ratio,
                                                  sinkhorn_scale=sinkhorn_scale,
                                                  gw_layout_coefficient=gw_layout_coefficient,
                                                  gw_layout=gw_layout,
                                                  sinkhorn_scale_gw=sinkhorn_scale_gw,
                                                  sinkhorn_iterations_gw=sinkhorn_iterations_gw)
        conditioners_list.append(conditioner)

    composed_cond = conditioners.ComposedConditioner(conditioners_list)
    return composed_cond


container_style=st.sidebar.container(border=True)
container_style.title('Display Style')

style=container_style.selectbox("Select Visualization Style:Can be 'stick', 'sphere', 'cross','cartoon'",('stick', 'sphere', 'cross','cartoon'),key='style')
resn=container_style.selectbox("Select the Amino Acid Type to Display",
                          ('*', 'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'),key='resn')
composeConditionerSampleDemo(style,resn)
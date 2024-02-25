import torch
from chroma import Chroma, Protein, conditioners, api
device = 'cuda' if torch.cuda.is_available() else 'cpu'
api.register_key(input("Enter API key: "))

# Initialize the Model
chroma = Chroma()

# Sample a Protein
protein = chroma.sample()
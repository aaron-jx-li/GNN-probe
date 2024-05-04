from huggingface_hub import login
import torch
import argparse
import json
import os
import time
import re
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"
login('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')
# model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'
model_name_or_path = 'meta-llama/Llama-2-13b-chat-hf'
# model_name_or_path = "EleutherAI/pythia-2.8b"
# model_name_or_path = "EleutherAI/pythia-1.4b"
# model_name_or_path = "EleutherAI/pythia-410m"
datafile_path = './balanced_world_place_masked.csv'
continent_to_label = {'Asia': 0, "Europe": 1, "North America": 2, "South America": 3, "Africa": 4, "Oceania": 5}

num_neighbors = 500

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, output_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = self.lin1(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        # x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        cache_dir = "/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models"
    )
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
         output_hidden_states=True,
        cache_dir = "/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models"
    ).to(device)

prompt = "Where is Stratford International station?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs,  output_hidden_states=True)
    hidden_states = outputs.hidden_states
output = model.generate(**inputs, max_new_tokens=100, output_hidden_states=True)
print("Trial response shape: ", output.shape)
print("Trial hidden state shape: ", hidden_states[1].shape)
print("Trial inference done")

feature_dim = hidden_states[1].shape[2]
print("Feature dimension: ", feature_dim)
datafile = pd.read_csv(datafile_path)
print("Number of nodes: ", len(datafile))
num_nodes = len(datafile)
features = torch.zeros((num_nodes, feature_dim))
coords = np.zeros((num_nodes, 2))
labels = torch.zeros(num_nodes, dtype=torch.long)
num_classes = len(continent_to_label.keys())

with torch.no_grad():
    for i in tqdm(range(len(datafile))):
        location_name = datafile.iloc[i]['name']
        prompt = f"Where is {location_name}?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs,  output_hidden_states=True)
        hidden_state = outputs.hidden_states[1][0][-1]
        assert len(hidden_state.shape) == 1 and hidden_state.shape[0] == feature_dim
        features[i, :] = hidden_state
        coords[i][0] = radians(float(datafile.iloc[i]['longitude'])) # x
        coords[i][1] = radians(float(datafile.iloc[i]['latitude'])) # y
        labels[i] = continent_to_label[datafile.iloc[i]['continent']]
        
print(features[0][:100])
print(coords[:20])
print(labels[:20])
print("Feature extraction done")

distances = haversine_distances(coords, coords)
print(distances.shape)
print("Distance calculation done")

nearest_neighbors = np.argsort(distances, axis=1)[:, 1:num_neighbors+1]  # Skip self (index 0)

edge_list = []
for i in range(num_nodes):
    for neighbor in nearest_neighbors[i]:
        edge_list.append((i, neighbor))
        edge_list.append((neighbor, i))

unique_edges = set(tuple(sorted(edge)) for edge in edge_list)

edge_index = torch.tensor(list(unique_edges), dtype=torch.long).t().contiguous()

# edge_indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
# # Convert list of tuples into a 2D Tensor
# edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

print(edge_index.shape)
print("Edge initialization done")

data = Data(x=features, edge_index=edge_index, y=labels).cuda(device)

model = GCN(input_dim=feature_dim, hidden_dim=512, output_dim=num_classes).cuda(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

data.train_mask = torch.tensor(datafile['is_test'] == False).cuda(device)
data.test_mask = torch.tensor(datafile['is_test'] == True).cuda(device)
print(data.train_mask[:20])
print(data.test_mask[:20])

def evaluate():
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc

accuracy = 0.0
for epoch in range(500):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss}')
    if epoch % 10 == 9:
        cur_accuracy = evaluate()
        print(f'Test Accuracy: {cur_accuracy}')
        if epoch > 50: 
            if cur_accuracy > accuracy:
                accuracy = cur_accuracy
            elif cur_accuracy > 0.94:
                break

        


# Llama2-13B: 0.01, 500: 0.951
# Llama3-8B: 0.01, 500: 0.924
# Pythia-2.8B: 0.01, 500: 0.827
# Pythia-1.4B: 0.01, 500: 0.726


# Extract embeddings for the test set
# test_embeddings = features[data.test_mask].numpy()
test_labels = data.y[data.test_mask].cpu().numpy()

# Apply t-SNE
model.eval()
tsne = TSNE(n_components=2)
with torch.no_grad():
    out = model(data.x, data.edge_index)
print(out.shape)
tsne_results = tsne.fit_transform(out[data.test_mask].detach().cpu().numpy())

# Plotting
plt.figure(figsize=(10, 6))
for class_label in range(num_classes):
    plt.scatter(tsne_results[test_labels == class_label, 0],
                tsne_results[test_labels == class_label, 1],
                label=list(continent_to_label.keys())[class_label], alpha=0.5)
# plt.title('t-SNE Visualization of Node Classifications')
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.legend()
plt.show()
plt.savefig('./llama2-13b.png')
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, auc
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Step 1: Load the Data
data_path = "/content/drive/MyDrive/GNN_for_EHR/data/eicu/storage_mimicv1/"
train_x, train_y = pickle.load(open(data_path + 'train_csr.pkl', 'rb'))
val_x, val_y = pickle.load(open(data_path + 'validation_csr.pkl', 'rb'))
test_x, test_y = pickle.load(open(data_path + 'test_csr.pkl', 'rb'))

# Step 2: Construct the Graph
def construct_graph(data):
    similarities = cosine_similarity(data)
    G = nx.Graph()
    for i in range(similarities.shape[0]):
        for j in range(i + 1, similarities.shape[1]):
            if similarities[i, j] > 0.85:
                G.add_edge(i, j, weight=similarities[i, j])
    return G

train_graph = construct_graph(train_x)
val_graph = construct_graph(val_x)
test_graph = construct_graph(test_x)

# Step 3: Identify Maximal Cliques
def maximal_cliques(G):
    return list(nx.find_cliques(G))

train_cliques = maximal_cliques(train_graph)
val_cliques = maximal_cliques(val_graph)
test_cliques = maximal_cliques(test_graph)

# Step 4: Define CliqueFluxNet Model
class CliqueFluxNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CliqueFluxNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)
        return torch.sigmoid(self.fc(x))

def edge_flux(graph, p_add=0.01, p_del=0.01):
    new_graph = graph.copy()
    num_edges = new_graph.number_of_edges()
    edges = list(new_graph.edges())
    
    # Randomly delete edges
    for edge in edges:
        if np.random.rand() < p_del:
            new_graph.remove_edge(*edge)
    
    # Randomly add edges
    nodes = list(new_graph.nodes())
    for _ in range(int(num_edges * p_add)):
        u, v = np.random.choice(nodes, 2)
        if not new_graph.has_edge(u, v):
            new_graph.add_edge(u, v)
    
    return new_graph

# Step 5: Train and Evaluate the Model
def train_model(train_graph, train_y, val_graph, val_y, model, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Apply Edge Flux
        train_graph_fluxed = edge_flux(train_graph)
        train_data = from_networkx(train_graph_fluxed)
        train_data.x = torch.tensor(train_x, dtype=torch.float)
        train_data.edge_index = train_data.edge_index
        
        output = model(train_data)
        loss = criterion(output, torch.tensor(train_y, dtype=torch.float))
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_data = from_networkx(val_graph)
            val_data.x = torch.tensor(val_x, dtype=torch.float)
            val_data.edge_index = val_data.edge_index
            val_output = model(val_data)
            val_loss = criterion(val_output, torch.tensor(val_y, dtype=torch.float))

            val_output = val_output.numpy()
            precision, recall, _ = precision_recall_curve(val_y, val_output)
            val_auprc = auc(recall, precision)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Val AUPRC: {val_auprc}')

input_dim = train_x.shape[1]
hidden_dim = 16
output_dim = 1
model = CliqueFluxNet(input_dim, hidden_dim, output_dim)

train_model(train_graph, train_y, val_graph, val_y, model)

# Step 6: Evaluate on Test Data
model.eval()
with torch.no_grad():
    test_data = from_networkx(test_graph)
    test_data.x = torch.tensor(test_x, dtype=torch.float)
    test_data.edge_index = test_data.edge_index
    test_output = model(test_data)

    test_output = test_output.numpy()
    precision, recall, _ = precision_recall_curve(test_y, test_output)
    test_auprc = auc(recall, precision)
    
    print(f'Test AUPRC: {test_auprc}')

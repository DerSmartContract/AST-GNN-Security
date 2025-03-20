################################################
# anomaly_detection_app.py
# ----------------------------------------------
# Ein einfaches Streamlit-Script, das:
# 1) Benutzerdefinierten Code als Eingabe nimmt
# 2) Python-AST erzeugt und als Graph darstellt
# 3) Ein sehr vereinfachtes GNN-Autoencoder-Setup
#    anwendet, um (fiktiv) einen Anomaliewert zu bestimmen.
################################################

import streamlit as st
import ast
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

################################################
# Hilfsfunktionen: AST -> NetworkX -> PyG
################################################

def parse_python_code_to_ast(code: str):
    """
    Parst Python-Code zu einem abstrakten Syntaxbaum (AST).
    Gibt das Wurzelknoten-Objekt des ASTs zurück.
    """
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError as e:
        st.error(f"SyntaxError beim Parsen: {e}")
        return None

def ast_to_nx_graph(ast_root):
    """
    Konvertiert einen Python-AST in einen NetworkX-Graphen.
    Jeder AST-Knoten wird ein Knoten im Graph,
    und Kanten werden entsprechend der Eltern-Kind-Beziehung eingefügt.
    """
    g = nx.DiGraph()

    def add_nodes_edges(node, parent_id=None):
        node_id = id(node)
        # Füge den Knoten hinzu, falls noch nicht vorhanden.
        if node_id not in g:
            g.add_node(node_id, nodetype=type(node).__name__)
        # Erstelle Kante zum Elternknoten (falls vorhanden).
        if parent_id is not None:
            g.add_edge(parent_id, node_id)
        # Gehe rekursiv durch die Felder des Knotens
        for field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                add_nodes_edges(field_value, node_id)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        add_nodes_edges(item, node_id)

    add_nodes_edges(ast_root, None)
    return g

def nx_to_pyg_data(nx_graph: nx.DiGraph) -> Data:
    """
    Wandelt NetworkX DiGraph in ein PyTorch-Geometric Data-Objekt um.
    """
    if nx_graph is None or len(nx_graph.nodes) == 0:
        return None

    # Knotenliste und Mapping erstellen
    nodes = list(nx_graph.nodes())
    node_idx_map = {node_id: i for i, node_id in enumerate(nodes)}
    
    # Kanten extrahieren
    edges = list(nx_graph.edges())
    if len(edges) == 0:
        # Minimale Fallunterscheidung: wenn keine Kanten vorhanden sind (z.B. leerer Code),
        # legen wir einen dummy-Eintrag an
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            [[node_idx_map[u], node_idx_map[v]] for u, v in edges],
            dtype=torch.long
        ).t().contiguous()
    
    # Node-Features (hier sehr rudimentär nur der nodetype als String → numeric encoding)
    node_types = [nx_graph.nodes[node_id]['nodetype'] for node_id in nodes]
    unique_types = list(set(node_types))
    type_to_idx = {typ: i for i, typ in enumerate(unique_types)}
    x = torch.tensor([type_to_idx[typ] for typ in node_types], dtype=torch.long).unsqueeze(1)
    
    data = Data(x=x, edge_index=edge_index)
    return data

################################################
# GNN-Autoencoder: stark vereinfachtes Modell
################################################
class SimpleGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class SimpleGCNDecoder(nn.Module):
    """
    Minimalbeispiel: versucht, die ursprünglichen Node-Features wiederherzustellen.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, z, edge_index):
        z = self.conv1(z, edge_index)
        z = F.relu(z)
        z = self.conv2(z, edge_index)
        return z

class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.encoder = SimpleGCNEncoder(in_channels, hidden_channels, latent_dim)
        self.decoder = SimpleGCNDecoder(latent_dim, hidden_channels, in_channels)
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z, edge_index)
        return x_hat, z

################################################
# Streamlit-Anwendung
################################################
def main():
    st.title("Einfaches AST-Anomalieerkennungs-Demo (Python-GNN)")
    
    st.write("""
    Dieses kleine Tool demonstriert, wie du Python-Code in einen AST umwandeln,
    ihn in einen Graph überführen und eine GNN-basierte (Autoencoder) 
    Anomalieerkennung vornehmen kannst.
    **Hinweis**: Das Modell ist untrainiert und dient nur als Beispiel.
    """)
    
    # Eingabebereich für benutzerdefinierten Code
    code_input = st.text_area(
        "Füge deinen Python-Code hier ein:",
        value="""def foo(x): 
    if x > 10:
        return x*2
    else:
        return x-1
""",
        height=200
    )
    
    if st.button("AST analysieren"):
        # 1) Parsen des Codes
        ast_root = parse_python_code_to_ast(code_input)
        if ast_root is None:
            st.warning("Kein AST erzeugt, Parsing fehlgeschlagen oder leerer Code.")
            return
        
        # 2) Konvertierung in NetworkX-Graphen
        nx_graph = ast_to_nx_graph(ast_root)
        st.success("AST wurde erfolgreich erzeugt und in NetworkX Graph konvertiert!")
        
        # 3) Ausgabe einiger Infos
        st.write(f"**Anzahl Knoten im Graph:** {nx_graph.number_of_nodes()}")
        st.write(f"**Anzahl Kanten im Graph:** {nx_graph.number_of_edges()}")
        
        # 4) Umwandlung in PyG Data
        pyg_data = nx_to_pyg_data(nx_graph)
        if pyg_data is None:
            st.warning("PyG-Konvertierung fehlgeschlagen oder leerer Graph.")
            return
        
        st.write(f"PyG Data Objekt: **{pyg_data}**")
        
        # Kurze Demo: GNN-Autoencoder (untrainiert!)
        # Setup des Modells (fixe Hyperparameter als Beispiel)
        in_channels = 1     # Weil wir (hier) nur eine einfache numerische Codierung haben
        hidden_dim = 8
        latent_dim = 4
        model = GraphAutoencoder(in_channels, hidden_dim, latent_dim)
        
        model.eval()
        with torch.no_grad():
            x_hat, z = model(pyg_data.x.float(), pyg_data.edge_index)
        
        # 5) Anomaliewert via Rekonstruktionsfehler
        # In der Praxis: Modell trainieren und dann diesen Error interpretieren
        criterion = nn.MSELoss()
        reconstruction_error = criterion(x_hat, pyg_data.x.float()).item()
        
        st.write(f"**Rekonstruktionsfehler (Dummy-Wert):** {reconstruction_error:.4f}")
        
        # Einfache Interpretation:
        # je größer, desto „ungewöhnlicher“ – in einer echten Anwendung 
        # brauchst du Schwellwerte basierend auf Trainingsdaten.
        if reconstruction_error > 1.0:
            st.error("Mögliche Anomalie entdeckt (Demo-Indikator > 1.0)!")
        else:
            st.success("Keine auffällige Anomalie (Demo-Indikator ≤ 1.0).")

if __name__ == "__main__":
    main()

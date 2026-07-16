import csv

# from rdkit.Chem import MolFromSmiles
import networkx as nx
import pandas as pd
from rdkit import Chem

from dataprocess.utils_test import *


def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set)) 

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

   
    c_size = mol.GetNumAtoms() 
    features = []  
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  
        features.append(feature / sum(feature))  

    
    edges = []  
    for bond in mol.GetBonds():  
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])  


    g = nx.Graph(edges).to_directed()  
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index  



def creat_data(datafile,drug_smiles_file, cellfile):

    
    file2 = cellfile
    cell_features = []
    with open(file2) as csvfile:
        csv_reader = csv.reader(csvfile)  
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)  
    #print('cell_features', cell_features)


    compound_iso_smiles = []
    df = pd.read_csv(drug_smiles_file)  
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)  
    smile_graph = {}
    #print('compound_iso_smiles', compound_iso_smiles)
    for smile in compound_iso_smiles:
        #print('smiles', smile)
        if smile not in smile_graph: 
            smile_graph[smile] = {}
        g = smile_to_graph(smile)  
        smile_graph[smile] = g 


    
    df = pd.read_csv(datafile)
    drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
    drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)
    # make dataprocess PyTorch Geometric ready

    return drug1, drug2, cell, label, smile_graph, cell_features



if __name__ == "__main__":

    cellfile = 'data/drugcombdb/dbcellfeature.csv'
    drug_smiles_file = 'data/drugcombdb/dbdrugsmiles.csv'
    datafile = 'data/drugcombdb/drugcombdb.csv'

    creat_data(datafile, drug_smiles_file, cellfile)

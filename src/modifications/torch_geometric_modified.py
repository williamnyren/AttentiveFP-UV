from typing import Any, Dict, List
import numpy as np
import torch

import torch_geometric
from torch_geometric.data import Data

x_map_original: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

x_map_one_hot: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': {
        'CHI_UNSPECIFIED':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'CHI_TETRAHEDRAL_CW':[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'CHI_TETRAHEDRAL_CCW':[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'CHI_OTHER':[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'CHI_TETRAHEDRAL':[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        'CHI_ALLENE':[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'CHI_SQUAREPLANAR':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        'CHI_TRIGONALBIPYRAMIDAL':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        'CHI_OCTAHEDRAL':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        },
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': {
        'UNSPECIFIED': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'S': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'SP': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'SP2': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        'SP3': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'SP3D': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        'SP3D2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        'OTHER': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        },
    'is_aromatic': {False: [0.0,1.0], True: [1.0,0.0]},
    'is_in_ring': {False:[0.0,1.0], True:[1.0,0.0]}
}
e_map_original: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

e_map_one_hot: Dict[str, List[Any]] = {
    'bond_type': {
        'UNSPECIFIED':[1.0,0.0,0.0,0.0,0.0,0.0,0.0],
        'SINGLE':[0.0,1.0,0.0,0.0,0.0,0.0,0.0],
        'DOUBLE':[0.0,0.0,1.0,0.0,0.0,0.0,0.0],
        'TRIPLE': [0.0,0.0,0.0,1.0,0.0,0.0,0.0],
        'OTHER':[0.0,0.0,0.0,0.0,1.0,0.0,0.0],
        'AROMATIC':[0.0,0.0,0.0,0.0,0.0,1.0,0.0],
        'LONGRANGE':[0.0,0.0,0.0,0.0,0.0,0.0,1.0],
        },
    # stereo in dataset seems to be of type STEREONONE for all molecules
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': {False:[0.0,1.0], True:[1.0,0.0]},
    'is_aromatic': {False:[0.0,1.0], True:[1.0,0.0]},
    'is_in_ring': {False:[0.0,1.0], True:[1.0,0.0]},
}
def node_encoding(mol, x_map, one_hot = False):
    xs = []
    if one_hot:
        for atom in mol.GetAtoms():
            row = []
            row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
            row += x_map['chirality'][str(atom.GetChiralTag())]
            row.append(x_map['degree'].index(atom.GetTotalDegree()))
            row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
            row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
            row.append(x_map['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()))
            row += x_map['hybridization'][str(atom.GetHybridization())]
            row += x_map['is_aromatic'][atom.GetIsAromatic()]
            row += x_map['is_in_ring'][atom.IsInRing()]
            xs.append(row)
        x = torch.tensor(xs, dtype=torch.float).view(-1, 26)
    else:
        for atom in mol.GetAtoms():
            row = []
            row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
            row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
            row.append(x_map['degree'].index(atom.GetTotalDegree()))
            row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
            row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
            row.append(x_map['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()))
            row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
            row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
            row.append(x_map['is_in_ring'].index(atom.IsInRing()))
            xs.append(row)
        x = torch.tensor(xs, dtype=torch.float).view(-1, 9)
    return x
def edge_encoding(mol, e_map, one_hot = False, add_fake_edges = False):
    num_atoms = mol.GetNumAtoms()
    bond_dict = {i : [] for i in range(num_atoms)}
    edge_indices, edge_attrs = [], []
    if one_hot:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_dict[i].append(j)
            e = []
            e += e_map['bond_type'][str(bond.GetBondType())]
            e.append(e_map['stereo'].index(str(bond.GetStereo())))
            e += e_map['is_conjugated'][bond.GetIsConjugated()]
            e += e_map['is_aromatic'][bond.GetIsAromatic()]
            e += e_map['is_in_ring'][bond.IsInRing()]

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]
        if add_fake_edges:
            selection = np.linspace(0, num_atoms, num=int(num_atoms*0.05), dtype=int, endpoint=False)
            for i in selection:
                for j in selection:
                    if j in bond_dict[i] or i == j:
                        continue
                    e = []
                    e += e_map['bond_type']['LONGRANGE']
                    e.append(6)
                    e += e_map['is_conjugated'][False]
                    e += e_map['is_aromatic'][False]
                    e += e_map['is_in_ring'][False]
                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e, e]
        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 14)
    else:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = []
            e.append(e_map['bond_type'].index(str(bond.GetBondType())))
            e.append(e_map['stereo'].index(str(bond.GetStereo())))
            e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]
        
        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 3)
        
    return edge_index, edge_attr


def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False, one_hot = False,
                add_fake_edges = False
                ) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)
    if one_hot:
        x_map = x_map_one_hot
    else:
        x_map = x_map_original
        
    x = node_encoding(mol, x_map, one_hot)

    if one_hot:
        e_map = e_map_one_hot
    else:
        e_map = e_map_original    
    edge_index, edge_attr = edge_encoding(mol, e_map, one_hot, add_fake_edges)
    

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, 
                edge_attr=edge_attr, smiles=smiles)



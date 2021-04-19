import torch
import torchani
import pandas as pd
import numpy as np
import plot_code as pto
import re
from torchani.units import HARTREE_TO_KCALMOL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_xyz(file):
    xyz = []
    typ = []
    Na = []
    ct = []
    fd = open(file, 'r').read()
    rb = re.compile('(\d+?)\n(.*?)\n((?:[A-Z][a-z]?.+?(?:\n|$))+)')
    ra = re.compile(
        '([A-Z][a-z]?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s*?(?:\n|$)'
    )
    s = rb.findall(fd)
    Nc = len(s)
    if Nc == 0:
        raise ValueError('No coordinates found in file. Check formatting of ' +
                         file + '.')
    for i in s:
        X = []
        T = []
        ct.append(i[1])
        c = ra.findall(i[2])
        Na.append(len(c))
        for j in c:
            T.append(j[0])
            X.append(j[1])
            X.append(j[2])
            X.append(j[3])
        X = np.array(X, dtype=np.float32)
        X = X.reshape(len(T), 3)
        xyz.append(X)
        typ.append(T)

    return xyz, typ, Na, ct

def calculaterootmeansqrerror(data1, data2, axis=0):
    data = np.power(data1 - data2, 2)
    return np.sqrt(np.mean(data, axis=axis))


def calculatemeanabserror(data1, data2, axis=0):
    data = np.abs(data1 - data2)
    return np.mean(data, axis=axis)


model = torchani.models.ANI2x(periodic_table_index=False).to(device)

species_to_tensor = torchani.utils.ChemicalSymbolsToInts(
    ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'])

molecules = ['cysteine_dipeptide', 'DDT', 'hexafluoroacetone', 'bendamustine']

for mol in molecules:
    fa = mol + '.xyz'
    X, S, Na, ct = read_xyz('xyzs/aniopt/' + fa)
    df = pd.read_csv(mol + '.csv')
    ani_energies = []
    for xi, si in zip(X, S):
        coordinates = torch.tensor([xi], requires_grad=True, device=device)
        species = species_to_tensor(si).unsqueeze(0).to(device)
        _, energy = model((species, coordinates))
        ani_energies.append(energy.item() * HARTREE_TO_KCALMOL)

    ani_energies = np.array(ani_energies) - min(ani_energies)
    dft_energies = np.array(df['DFT energy']) - min(df['DFT energy'])

    MAE = calculatemeanabserror(ani_energies, dft_energies)
    RMSE = calculaterootmeansqrerror(ani_energies, dft_energies)
    print(mol)
    print(mol, 'MAE :', MAE)
    print(mol, 'RMSE :', RMSE)
    phi = range(36)
    psi = range(36)
    phi = np.array(phi)
    psi = np.array(psi)
    phi = phi * 10 - 180.
    psi = psi * 10 - 180.
    Z = [ani_energies, dft_energies]
    pto.plot_2d(phi, psi, Z, xax='PHI', yax='PSI', save=mol)

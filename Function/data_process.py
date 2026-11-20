
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def load_and_process_data(file_path, num_zero_threshold):

    df = pd.read_csv(file_path)

    
    df = df.dropna(subset=["Smiles", "Eg"]).copy()

    # Generate molecules
    molecules = df["Smiles"].apply(Chem.MolFromSmiles)

    # Remove invalid SMILES
    valid_mask = molecules.notna()

    molecules = molecules[valid_mask]
    df = df.loc[valid_mask]

    # Generate MFP
    fp = molecules.apply(
        lambda m: AllChem.GetMorganFingerprint(m, radius=3)
    )

    fp_n = fp.apply(lambda m: m.GetNonzeroElements())

    # FP intenifier
    HashCode = []

    for i in fp_n:
        for j in i.keys():
            HashCode.append(j)

    unique_set = set(HashCode)
    unique_list = list(unique_set)

    # Create fingerprint index mapping
    hash_to_index = {
        key: index
        for index, key in enumerate(unique_list)
    }

    # Generate fingerprint matrix
    MY_finger = []

    for polymer in fp_n:

        my_finger = [0] * len(unique_list)

        for key in polymer.keys():

            index = hash_to_index[key]

            my_finger[index] = polymer[key]

        MY_finger.append(my_finger)

    X = pd.DataFrame(MY_finger)

    # Remove rare fingerprint features
    Zero_Sum = (X == 0).sum()

    Columns = Zero_Sum[
        Zero_Sum <= (len(X) - num_zero_threshold)
    ].index

    X_count = X[Columns]

    # Target variable
    Y = df["Eg"].to_numpy(dtype=np.float32)

    return X_count, Y

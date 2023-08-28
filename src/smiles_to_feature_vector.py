import os
import pandas as pd
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence

dirname = os.path.dirname(__file__)
dirname = dirname.replace("\\", "/")
dirname += "/"


def _generate_mol_feature_vector(dataset_row, model, threshold):
    # count the nr of identifiers or substructures which are found in the pretrained model
    all_identifiers = mol2alt_sentence(Chem.MolFromSmiles(dataset_row["smiles"]), 1)
    existing_identifiers = [id for id in all_identifiers if id in model.wv]
    ratio_existing = len(existing_identifiers) / len(all_identifiers)
    if ratio_existing < threshold:
        return None

    # get_mean_vector returns the mean of all the vectors for the given identifiers
    # return vector as pd.DataFrame with one row
    return pd.DataFrame(model.wv.get_mean_vector(existing_identifiers)).T


def generate_feature_vectors_from_smiles(
    dataset: pd.DataFrame, existing_ids_threshold: float = 1.0
) -> dict[int, pd.DataFrame]:
    """
    @param dataset: Must be a Pandas DataFrame containing a column 'smiles' with SMILES representation of molecules
    @param existing_ids_threshold: Minimum ratio of substructures per molecule that must exist in the dictionary.
        E.g. if existing_ids_threshold=0.9 and a molecule has 100 substructures, but for 15 substructures the model
        cannot find a corresponding feature vector (85% < 90%), then the molecule is discarded

    @returns: dict of molecule feature vectors that had enough existing identifiers
        the keys is the index of the molecule in the original dataset
        the value is a (1x100) pd.DataFrame where each column is one dimension of the feature vector
    """
    model = word2vec.Word2Vec.load(dirname + "../../models/model_100dim.pkl")
    mol_feature_vecs = {
        idx: _generate_mol_feature_vector(row, model, existing_ids_threshold)
        for idx, row in dataset.iterrows()
    }
    return {
        idx: mol for idx, mol in mol_feature_vecs.items() if mol is not None
    }  # filter out None entries


def main():
    df = pd.read_csv(dirname + "../../data/raw/esol.csv")
    print(df.head())
    transformed = generate_feature_vectors_from_smiles(df.loc[:99, :])[0]
    print(transformed.head())


if __name__ == "__main__":
    main()

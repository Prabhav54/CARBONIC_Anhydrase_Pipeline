import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_fingerprints(self, smiles_list, n_bits=2048):
        """
        Custom function to convert SMILES to Morgan Fingerprints.
        """
        fps = []
        valid_indices = []
        logging.info("Generating ECFP4 Fingerprints...")
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                    fps.append(np.array(fp))
                    valid_indices.append(i)
            except:
                continue
        return np.array(fps), valid_indices

    def get_data_transformer_object(self):
        """
        This function creates the transformation pipeline for Protein Features.
        (Fingerprints don't need scaling, so we handle them separately).
        """
        try:
            # We only scale the continuous protein features
            numerical_columns = [
                'Protein_Weight', 
                'Protein_Aromaticity', 
                'Protein_Isoelectric', 
                'Protein_Hydrophobicity'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns for scaling: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ],
                remainder="passthrough" # Keep other columns (like Fingerprints) as is
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Generating Fingerprints for Training Data...")
            # 1. Generate Fingerprints (Train)
            train_fps, train_idx = self.get_fingerprints(train_df['clean_smiles'])
            train_fps_df = pd.DataFrame(train_fps, columns=[f'fp_{i}' for i in range(2048)])
            
            # 2. Extract Protein Features (Train)
            protein_cols = ['Protein_Weight', 'Protein_Aromaticity', 'Protein_Isoelectric', 'Protein_Hydrophobicity']
            train_protein = train_df.iloc[train_idx][protein_cols].reset_index(drop=True)
            
            # 3. Combine Features (Train)
            input_feature_train_df = pd.concat([train_protein, train_fps_df], axis=1)
            target_feature_train_df = train_df.iloc[train_idx]['pIC50'].reset_index(drop=True)

            logging.info("Generating Fingerprints for Testing Data...")
            # 4. Generate Fingerprints (Test)
            test_fps, test_idx = self.get_fingerprints(test_df['clean_smiles'])
            test_fps_df = pd.DataFrame(test_fps, columns=[f'fp_{i}' for i in range(2048)])
            
            # 5. Extract Protein Features (Test)
            test_protein = test_df.iloc[test_idx][protein_cols].reset_index(drop=True)
            
            # 6. Combine Features (Test)
            input_feature_test_df = pd.concat([test_protein, test_fps_df], axis=1)
            target_feature_test_df = test_df.iloc[test_idx]['pIC50'].reset_index(drop=True)

            logging.info("Applying Preprocessing Object on training and testing dataframes")
            
            preprocessing_obj = self.get_data_transformer_object()

            # Note: We fit on the protein columns inside input_feature_train_df
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate X and Y for the Trainer
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
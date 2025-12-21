import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a pipeline that handles missing data and scales numbers.
        """
        try:
            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # Fill missing values
                    ("scaler", StandardScaler())                   # Scale to mean=0, std=1
                ]
            )
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_data(self, df):
        """
        Cleans data and selects the correct columns for Structure-Based Training.
        """
        try:
            # 1. Filter Invalid Entries
            # We need a valid IC50 value to calculate the target
            df = df.dropna(subset=['standard_value'])
            df = df[df['standard_value'] > 0]
            
            # 2. Engineer Target: Convert IC50 (nM) -> pIC50
            # Formula: pIC50 = 9 - log10(IC50_nM)
            # Higher pIC50 = More Potent
            df['pIC50'] = 9 - np.log10(df['standard_value'])
            
            # Remove infinities created by log(0) errors
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pIC50'])
            
            # 3. Drop Metadata Columns
            # We remove names so the model learns from *Physics*, not memorization.
            # We drop 'target_name' specifically because we want the model to look at 'Protein_Hydrophobicity', not the name "CA-II".
            cols_to_drop = [
                'molecule_chembl_id', 
                'target_chembl_id', 
                'target_name',       
                'smiles', 
                'standard_value', 
                'standard_units', 
                'pIC50',             # Target (separated later)
                'bioactivity_type',
                'units',
                'value',
                'bao_label',
                'standard_type',
                'standard_relation',
                'document_chembl_id'
            ]
            
            # Dynamically select all numeric features (Protein_* + Drug Features)
            feature_cols = [col for col in df.columns if col not in cols_to_drop and df[col].dtype != 'O']
            
            X = df[feature_cols]
            y = df['pIC50']
            
            return X, y, feature_cols

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting Data Transformation...")
            
            # 1. Read the Train/Test files (Created by Ingestion in Phase 1)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Loaded Train Data: {train_df.shape}")
            logging.info(f"Loaded Test Data: {test_df.shape}")

            # 2. Clean & Select Features
            input_feature_train_df, target_feature_train_df, cols = self.prepare_data(train_df)
            input_feature_test_df, target_feature_test_df, _ = self.prepare_data(test_df)
            
            logging.info(f"Selected {len(cols)} Features for Training (Protein + Ligand).")

            # 3. Apply Scaling
            # CRITICAL: Fit ONLY on Training data to avoid data leakage
            preprocessing_obj = self.get_data_transformer_object()
            
            logging.info("Fitting Scaler on Training Data...")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 4. Combine X and y into single arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 5. Save the Preprocessor (So we can reuse it for prediction later)
            save_object(self.config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info(f"Saved Preprocessor to {self.config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
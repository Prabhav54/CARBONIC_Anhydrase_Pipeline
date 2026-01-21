import sys
import pandas as pd
import os

# Add src to path so imports work correctly
sys.path.append(os.getcwd())

from src.pipeline.predict_pipeline import VirtualScreeningPipeline, PredictPipeline

def main():
    print("="*50)
    print("   AI DRUG DISCOVERY PIPELINE START")
    print("="*50)

    try:
        # --- STEP 1: ML VIRTUAL SCREENING ---
        # Initialize the ML screening pipeline
        ml_screener = VirtualScreeningPipeline()
        
        print("\n[PHASE 1] ML-Based Screening (pIC50 Prediction)...")
        # Run screening for Carbonic Anhydrase (3HS4)
        top_candidates_df = ml_screener.run_screening(pdb_id="3HS4")
        
        print(f"✅ Screening Complete. Identified {len(top_candidates_df)} top candidates.")
        print(top_candidates_df[['clean_smiles', 'Predicted_pIC50', 'IC50_nM']].head())

        # --- STEP 2: PREPARE FOR PHYSICS PIPELINE ---
        # Convert DataFrame to the list-of-dicts format required by PredictPipeline
        physics_candidates = []
        for index, row in top_candidates_df.iterrows():
            physics_candidates.append({
                'name': f"Rank_{index+1}_pIC50_{row['Predicted_pIC50']:.2f}",
                'smiles': row['clean_smiles']
            })

        # --- STEP 3: PHYSICS VALIDATION (ADMET -> Docking -> MD) ---
        physics_pipeline = PredictPipeline()
        
        print("\n[PHASE 2] Physics-Based Validation (ADMET, Docking, MD)...")
        final_results_df = physics_pipeline.run(physics_candidates)

        # --- STEP 4: REPORTING ---
        print("\n" + "="*50)
        print("   FINAL PIPELINE REPORT")
        print("="*50)
        
        # Merge ML predictions with Physics results
        # We do a simple merge on 'smiles' to keep all data together
        if not final_results_df.empty:
            merged_report = pd.merge(
                final_results_df, 
                top_candidates_df[['clean_smiles', 'Predicted_pIC50', 'Status']], 
                left_on='smiles', 
                right_on='clean_smiles', 
                how='left'
            )
            
            # Select relevant columns for display
            cols_to_show = [
                'name', 'Status_x', 'Predicted_pIC50', 'Docking_Score', 
                'MD_Status', 'Violations', 'QED'
            ]
            # Handle potential missing columns if run failed
            available_cols = [c for c in cols_to_show if c in merged_report.columns]
            
            print(merged_report[available_cols])
            
            # Save to CSV
            merged_report.to_csv("artifacts/final_drug_discovery_report.csv", index=False)
            print(f"\n✅ Full report saved to: artifacts/final_drug_discovery_report.csv")
        else:
            print("❌ No candidates passed the pipeline.")

    except Exception as e:
        print(f"\n❌ PIPELINE CRASHED: {str(e)}")
        # Optional: Print traceback for debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs
import requests
import molmass
import io

# ===================================================================
# 1. CONFIG & HELPER FUNCTIONS
# ===================================================================
st.set_page_config(page_title="Metabolomics Merger", layout="wide")

def get_formula_from_smiles(smiles):
    """Replaces R 'custom_formula' using RDKit. Faster, no OpenBabel needed."""
    try:
        if pd.isna(smiles) or smiles == "": return None
        mol = Chem.MolFromSmiles(smiles)
        if mol: return rdMolDescriptors.CalcMolFormula(mol)
        return None
    except: return None

def calculate_tanimoto(smiles1, smiles2):
    """Replaces R 'calculate_tanimoto...'. Uses Morgan fingerprints (ECFP4-like)."""
    try:
        if pd.isna(smiles1) or pd.isna(smiles2) or smiles1 == "" or smiles2 == "": return np.nan
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 and mol2:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        return np.nan
    except: return np.nan

def get_accurate_mass(formula, adduct):
    """Calculates exact m/z for a given formula and adduct."""
    try:
        if pd.isna(formula) or pd.isna(adduct) or formula == "": return np.nan
        # Create molmass object
        f = molmass.Formula(formula)
        
        # Simple adduct handling (expand this list if needed based on your data)
        adduct_masses = {
            "[M+H]+": 1.007276, "[M]+": 0.0, "[M+Na]+": 22.98977, 
            "[M+NH4]+": 18.03383, "[M+K]+": 38.96370, "[M+2H]2+": 1.007276*2, # simplified
             "[M-H]-": -1.007276, "[M+Cl]-": 34.96885
        }
        
        # Handle multiply charged species approximately if not strictly defined
        charge = 1
        if "2+" in adduct: charge = 2
        elif "3+" in adduct: charge = 3
        
        mass_shift = adduct_masses.get(adduct, 0.0)
        # If adduct not in simple list, might need more complex parsing, 
        # but this covers 95% of standard FBMN outputs.
        
        return (f.isotope.mass + mass_shift) / charge
    except: return np.nan

def get_npclassifier_data(smiles):
    """Queries GNPS2 NPClassifier API."""
    if pd.isna(smiles) or smiles == "": return np.nan, np.nan, np.nan
    try:
        url = f"https://npclassifier.gnps2.org/classify?smiles={requests.utils.quote(smiles)}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            d = response.json()
            return (";".join(d.get('pathway_results', [])), 
                    ";".join(d.get('superclass_results', [])), 
                    ";".join(d.get('class_results', [])))
    except: pass
    return np.nan, np.nan, np.nan

# ===================================================================
# 2. MAIN APP INTERFACE
# ===================================================================
st.title("🧬 Metabolomics Tiered Merger")
st.markdown("Upload outputs from FBMN, MolDiscovery, Dereplicator+, and SIRIUS to merge them using tiered confidence logic.")

with st.expander("📁 File Uploads", expanded=True):
    c1, c2 = st.columns(2)
    fbmn_file = c1.file_uploader("FBMN Global Table (.tsv)", type=['tsv', 'txt'])
    ms1_file = c1.file_uploader("Molecular Network / MS1 (.csv)", type=['csv'])
    sirius_file = c1.file_uploader("SIRIUS Results (.csv)", type=['csv', 'txt'])
    moldisc_file = c2.file_uploader("MolDiscovery (.tsv)", type=['tsv', 'txt'])
    derep_file = c2.file_uploader("Dereplicator+ (.tsv)", type=['tsv', 'txt'])

# ===================================================================
# 3. PIPELINE LOGIC
# ===================================================================
if st.button("🚀 Run Merging Pipeline", type="primary"):
    if not (fbmn_file and ms1_file):
        st.error("Missing required files: FBMN and MS1 are mandatory.")
        st.stop()

    with st.status("Processing...", expanded=True) as status:
        # --- A. LOAD & STANDARDIZE ---
        st.write("Loading and standardizing dataframes...")
        
        # MS1
        ms1 = pd.read_csv(ms1_file)
        ms1['feature'] = ms1['precursor mass'].astype(str) + "_" + ms1['RTMean'].astype(str)
        ms1 = ms1.rename(columns={'cluster index': 'row.ID', 'precursor mass': 'row.m.z', 'RTMean': 'row.retention.time'})
        ms1 = ms1[['row.ID', 'row.m.z', 'row.retention.time', 'feature']]

        # FBMN
        fbmn = pd.read_csv(fbmn_file, sep='\t')
        # Fix #Scan# issue specifically requested
        if '#Scan#' in fbmn.columns: fbmn = fbmn.rename(columns={'#Scan#': 'row.ID'})
        elif 'Scan' in fbmn.columns: fbmn = fbmn.rename(columns={'Scan': 'row.ID'})
        
        # Normalize Adducts common in FBMN
        adduct_map = {"M+H":"[M+H]+", "M+2H":"[M+2H]2+", "M+Na":"[M+Na]+", "M+NH4":"[M+NH4]+", "M-H2O+H":"[M+H-H2O]+"}
        if 'Adduct' in fbmn.columns: fbmn['Adduct'] = fbmn['Adduct'].replace(adduct_map)

        # Load optional files with IMMEDIATE suffixes to match your debug output
        mold = pd.read_csv(moldisc_file, sep='\t').add_suffix('_mold').rename(columns={'#Scan#_mold':'row.ID', 'Scan_mold':'row.ID'}) if moldisc_file else pd.DataFrame(columns=['row.ID'])
        derep = pd.read_csv(derep_file, sep='\t').add_suffix('_derep').rename(columns={'#Scan#_derep':'row.ID', 'Scan_derep':'row.ID'}) if derep_file else pd.DataFrame(columns=['row.ID'])
        # Handle potential SIRIUS ID column variations before adding suffix
        if sirius_file:
             s_temp = pd.read_csv(sirius_file)
             # Try to find the ID column before suffixing
             for id_col in ['ID_extract', 'id', 'scan', 'compoundid']:
                 if id_col in s_temp.columns:
                     s_temp = s_temp.rename(columns={id_col: 'row.ID'})
                     break
             sirius = s_temp.add_suffix('_sirius').rename(columns={'row.ID_sirius': 'row.ID'})
        else:
             sirius = pd.DataFrame(columns=['row.ID'])

        # --- B. MERGE ---
        st.write("Merging tables...")
        # Use outer join on row.ID to ensure we don't lose features if FBMN is filtered
        df = pd.merge(fbmn, ms1, on='row.ID', how='outer')
        if not mold.empty: df = pd.merge(df, mold, on='row.ID', how='left')
        if not derep.empty: df = pd.merge(df, derep, on='row.ID', how='left')
        if not sirius.empty: df = pd.merge(df, sirius, on='row.ID', how='left')

        # Ensure numeric types for critical score columns
        cols_to_numeric = ['MQScore', 'FDR_mold', 'Score_mold', 'ConfidenceScoreExact_sirius', 'FDR_derep', 'row.m.z']
        for col in cols_to_numeric:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- C. CALCULATE SIMILARITIES ---
        if 'SMILES_mold' in df.columns and 'smiles_sirius' in df.columns:
            st.write("Calculating Tanimoto similarities (MolDiscovery vs SIRIUS)...")
            # Using apply here as it's hard to vectorize RDKit objects efficiently without generic numpy arrays
            df['Tanimoto_molD_Sirius'] = df.apply(lambda x: calculate_tanimoto(x.get('SMILES_mold'), x.get('smiles_sirius')), axis=1)
        else:
            df['Tanimoto_molD_Sirius'] = np.nan

        # --- D. TIERED ANNOTATION LOGIC ---
        st.write("Applying tiered annotation rules...")

        # 1. SAFEGUARD: Ensure all logic columns exist, even if files were not uploaded.
        # This prevents 'NoneType' errors in the conditions below.
        required_cols = [
            'MQScore', 'FDR_mold', 'ConfidenceScoreExact_sirius', 
            'Tanimoto_molD_Sirius', 'row.m.z', 'FDR_derep', 
            'NPC#pathway_sirius', 'molecularFormula.y_sirius', 'molecularFormula.x_sirius'
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        # 2. Define conditions using standard pandas comparisons (NaNs automatically evaluate to False in these checks)
        c_gnps = (df['MQScore'] > 0)
        c_consensus = (df['FDR_mold'] < 1) & (df['ConfidenceScoreExact_sirius'] > 0.6) & (df['Tanimoto_molD_Sirius'] > 0.85)
        c_mold_high = (df['row.m.z'] > 600) & (df['FDR_mold'] < 1)
        c_sirius_high = (df['row.m.z'] <= 600) & (df['ConfidenceScoreExact_sirius'] > 0.6)
        c_derep = (df['FDR_derep'] <= 5)
        c_mold_low = (df['FDR_mold'] < 5)
        c_sirius_low = (df['ConfidenceScoreExact_sirius'] > 0.1)
        c_canopus = df['NPC#pathway_sirius'].notna()

        conditions = [c_gnps, c_consensus, c_mold_high, c_sirius_high, c_derep, c_mold_low, c_sirius_low, c_canopus]
        choices = [
            "GNPS/Spectral Match", "Consensus (molD + Sirius)", "molDiscovery (>600 Da)",
            "Sirius/CSI:FingerID (<=600 Da)", "Dereplicator+", "molDiscovery (fallback)",
            "Sirius/CSI:FingerID (fallback)", "Sirius/CANOPUS"
        ]
        
        # np.select is now safe because all conditions are guaranteed to be boolean Series of the same length
        df['Annotated'] = np.select(conditions, choices, default=np.nan)

        # --- E. CONSOLIDATE FINAL COLUMNS ---
        # (Using .get() here is still fine as these are just string retrievals, not boolean logic tests)
        df['Final_Name'] = np.select(conditions, [
            df.get('Compound_Name'), df.get('Name_mold'), df.get('Name_mold'), 
            df.get('name_sirius'), df.get('Name_derep'), df.get('Name_mold'), 
            df.get('name_sirius'), "MolecularFormula_Class_predicted"
        ], default=np.nan)

        df['Final_SMILES'] = np.select(conditions, [
            df.get('Smiles'), df.get('SMILES_mold'), df.get('SMILES_mold'), 
            df.get('smiles_sirius'), df.get('SMILES_derep'), df.get('SMILES_mold'), 
            df.get('smiles_sirius'), np.nan
        ], default=np.nan)

        df['Final_Adduct'] = np.select(conditions, [
            df.get('Adduct'), df.get('Adduct_mold'), df.get('Adduct_mold'),
            df.get('adduct.y_sirius'), df.get('Adduct_derep'), df.get('Adduct_mold'),
            df.get('adduct.y_sirius'), df.get('adduct.x_sirius')
        ], default=np.nan)

        # Final Formula Logic
        df['Final_Formula'] = np.select(
            [c_sirius_high | c_sirius_low, c_canopus], 
            [df['molecularFormula.y_sirius'], df['molecularFormula.x_sirius']], 
            default=np.nan
        )
        # Fill remaining gaps
        mask_need_formula = df['Final_Formula'].isna() & df['Final_SMILES'].notna()
        if mask_need_formula.any():
             df.loc[mask_need_formula, 'Final_Formula'] = df[mask_need_formula]['Final_SMILES'].apply(get_formula_from_smiles)

        # --- E. CONSOLIDATE FINAL COLUMNS ---
        # 1. Final Name
        df['Final_Name'] = np.select(conditions, [
            df.get('Compound_Name'), df.get('Name_mold'), df.get('Name_mold'), 
            df.get('name_sirius'), df.get('Name_derep'), df.get('Name_mold'), 
            df.get('name_sirius'), "MolecularFormula_Class_predicted"
        ], default=np.nan)

        # 2. Final SMILES
        df['Final_SMILES'] = np.select(conditions, [
            df.get('Smiles'), df.get('SMILES_mold'), df.get('SMILES_mold'), 
            df.get('smiles_sirius'), df.get('SMILES_derep'), df.get('SMILES_mold'), 
            df.get('smiles_sirius'), np.nan
        ], default=np.nan)

        # 3. Final Formula (Complex: tries SIRIUS first, then calcs from SMILES)
        # First, get best available formula from SIRIUS depending on tier
        formula_sirius_y = df.get('molecularFormula.y_sirius', pd.Series([np.nan]*len(df)))
        formula_sirius_x = df.get('molecularFormula.x_sirius', pd.Series([np.nan]*len(df)))
        
        df['Final_Formula'] = np.select(
            [c_sirius_high | c_sirius_low, c_canopus], 
            [formula_sirius_y, formula_sirius_x], 
            default=np.nan
        )
        # Fill remaining gaps by calculating from Final_SMILES
        mask_need_formula = df['Final_Formula'].isna() & df['Final_SMILES'].notna()
        if mask_need_formula.any():
             df.loc[mask_need_formula, 'Final_Formula'] = df[mask_need_formula]['Final_SMILES'].apply(get_formula_from_smiles)

        # 4. Final Adduct
        df['Final_Adduct'] = np.select(conditions, [
            df.get('Adduct'), df.get('Adduct_mold'), df.get('Adduct_mold'),
            df.get('adduct.y_sirius'), df.get('Adduct_derep'), df.get('Adduct_mold'),
            df.get('adduct.y_sirius'), df.get('adduct.x_sirius')
        ], default=np.nan)

        # --- F. ACCURACY & NPCLASSIFIER ---
        # Filter to only annotated rows for expensive API calls
        df_final = df.dropna(subset=['Annotated']).copy()
        
        st.write(f"Found {len(df_final)} annotated features. Fetching NPClassifier classes...")
        
        # Accurate Mass
        df_final['Accurate_Mass'] = df_final.apply(lambda x: get_accurate_mass(x.get('Final_Formula'), x.get('Final_Adduct')), axis=1)
        # custom_ppm inline
        df_final['ppm_error'] = ((df_final['row.m.z'] - df_final['Accurate_Mass']) / df_final['Accurate_Mass']) * 1000000

        # NPClassifier Loop (with progress bar)
        # Pre-fill with SIRIUS NPC results if available (saves API calls)
        df_final['NPC_Pathway'] = df_final.get('NPC#pathway_sirius')
        df_final['NPC_Superclass'] = df_final.get('NPC#superclass_sirius')
        df_final['NPC_Class'] = df_final.get('NPC#class_sirius')

        # Identify rows that still need NPC from API (have SMILES but no SIRIUS NPC)
        mask_api_needed = df_final['Final_SMILES'].notna() & df_final['NPC_Pathway'].isna()
        rows_to_query = df_final[mask_api_needed]
        
        if not rows_to_query.empty:
            progress_bar = st.progress(0)
            total = len(rows_to_query)
            for i, (idx, row) in enumerate(rows_to_query.iterrows()):
                p, s, c_cls = get_npclassifier_data(row['Final_SMILES'])
                df_final.at[idx, 'NPC_Pathway'] = p
                df_final.at[idx, 'NPC_Superclass'] = s
                df_final.at[idx, 'NPC_Class'] = c_cls
                if i % 5 == 0: progress_bar.progress(min(1.0, (i + 1) / total))
            progress_bar.empty()

        status.update(label="Merging Complete!", state="complete", expanded=False)

    # --- G. RESULTS & DOWNLOAD ---
    st.success(f"Pipeline finished! {len(df_final)} features annotated.")
    
    # Final column selection for clean output
    final_cols = [
        'row.ID', 'row.m.z', 'row.retention.time', 'Annotated', 'Final_Name', 
        'Final_Formula', 'Final_Adduct', 'ppm_error', 'NPC_Pathway', 
        'NPC_Superclass', 'NPC_Class', 'Final_SMILES'
    ]
    # Only keep columns that actually exist
    existing_cols = [c for c in final_cols if c in df_final.columns]
    st.dataframe(df_final[existing_cols].head(20))

    # CSV Download
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Merged Annotation Table",
        csv,
        "Merged_Annotations.csv",
        "text/csv",
        key='download-csv'
    )
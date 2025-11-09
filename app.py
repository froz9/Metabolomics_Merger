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
st.set_page_config(page_title="Metabolomics Tiered Merger", layout="wide", page_icon="🧬")

def get_formula_from_smiles(smiles):
    """Replaces R 'custom_formula' using RDKit."""
    try:
        if pd.isna(smiles) or smiles == "": return None
        mol = Chem.MolFromSmiles(smiles)
        if mol: return rdMolDescriptors.CalcMolFormula(mol)
        return None
    except: return None

def calculate_tanimoto(smiles1, smiles2):
    """Replaces R 'calculate_tanimoto...'. Uses Morgan fingerprints (ECFP4)."""
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
        f = molmass.Formula(formula)
        adduct_masses = {
            "[M+H]+": 1.007276, "[M]+": 0.0, "[M+Na]+": 22.98977, 
            "[M+NH4]+": 18.03383, "[M+K]+": 38.96370, "[M+2H]2+": 1.007276*2,
             "[M-H]-": -1.007276, "[M+Cl]-": 34.96885, "[M+H-H2O]+": 1.007276 - 18.010565
        }
        charge = 2 if "2+" in adduct else (3 if "3+" in adduct else 1)
        mass_shift = adduct_masses.get(adduct, 0.0)
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
st.markdown("""
Upload your outputs from **FBMN**, **MolDiscovery**, **Dereplicator+**, and **SIRIUS**. 
This tool merges them using a tiered confidence approach based on community standards.
""")

with st.expander("📁 File Uploads (Drag and Drop)", expanded=True):
    c1, c2 = st.columns(2)
    fbmn_file = c1.file_uploader("FBMN Global Table (.tsv)", type=['tsv', 'txt'], help="Usually has 'Scan' or '#Scan#' column.")
    ms1_file = c1.file_uploader("Molecular Network / MS1 (.csv)", type=['csv'], help="Contains 'cluster index', 'precursor mass', 'RTMean'.")
    sirius_file = c1.file_uploader("SIRIUS Results (.csv)", type=['csv', 'txt'], help="Optional. Adds high-confidence in-silico annotations.")
    moldisc_file = c2.file_uploader("MolDiscovery (.tsv)", type=['tsv', 'txt'], help="Optional. Great for small peptides/natural products.")
    derep_file = c2.file_uploader("Dereplicator+ (.tsv)", type=['tsv', 'txt'], help="Optional. Peptidic annotation tool.")

# ===================================================================
# 3. PIPELINE LOGIC
# ===================================================================
if st.button("🚀 Run Merging Pipeline", type="primary"):
    if not (fbmn_file and ms1_file):
        st.error("❌ Missing required files: FBMN and MS1 are mandatory to start.")
        st.stop()

    with st.status("🔄 Processing Pipeline...", expanded=True) as status:
        
        # --- A. LOAD & STANDARDIZE (Using fixes from local_debug.py) ---
        st.write("📝 Loading and standardizing dataframes...")
        
        # MS1
        ms1 = pd.read_csv(ms1_file)
        ms1 = ms1.rename(columns={'cluster index': 'row.ID', 'precursor mass': 'row.m.z', 'RTMean': 'row.retention.time'})
        ms1['row.ID'] = ms1['row.ID'].astype(str) # FIX: Force String ID
        ms1 = ms1[['row.ID', 'row.m.z', 'row.retention.time']]

        # FBMN
        fbmn = pd.read_csv(fbmn_file, sep='\t')
        if '#Scan#' in fbmn.columns: fbmn = fbmn.rename(columns={'#Scan#': 'row.ID'})
        elif 'Scan' in fbmn.columns: fbmn = fbmn.rename(columns={'Scan': 'row.ID'})
        fbmn['row.ID'] = fbmn['row.ID'].astype(str) # FIX: Force String ID
        
        adduct_map = {"M+H":"[M+H]+", "M+2H":"[M+2H]2+", "M+Na":"[M+Na]+", "M+NH4":"[M+NH4]+", "M-H2O+H":"[M+H-H2O]+", "M+K":"[M+K]+"}
        if 'Adduct' in fbmn.columns: fbmn['Adduct'] = fbmn['Adduct'].replace(adduct_map)

        # Optional Files
        mold = pd.read_csv(moldisc_file, sep='\t').add_suffix('_mold').rename(columns={'#Scan#_mold':'row.ID', 'Scan_mold':'row.ID'}) if moldisc_file else pd.DataFrame(columns=['row.ID'])
        if not mold.empty: mold['row.ID'] = mold['row.ID'].astype(str)

        derep = pd.read_csv(derep_file, sep='\t').add_suffix('_derep').rename(columns={'#Scan#_derep':'row.ID', 'Scan_derep':'row.ID'}) if derep_file else pd.DataFrame(columns=['row.ID'])
        if not derep.empty: derep['row.ID'] = derep['row.ID'].astype(str)

        if sirius_file:
             s_temp = pd.read_csv(sirius_file)
             for id_col in ['ID_extract', 'id', 'scan', 'compoundid']:
                 if id_col in s_temp.columns:
                     s_temp = s_temp.rename(columns={id_col: 'row.ID'})
                     break
             sirius = s_temp.add_suffix('_sirius').rename(columns={'row.ID_sirius': 'row.ID'})
             sirius['row.ID'] = sirius['row.ID'].astype(str)
        else:
             sirius = pd.DataFrame(columns=['row.ID'])

        # --- B. MERGE ---
        st.write("🔗 Merging tables...")
        df = pd.merge(fbmn, ms1, on='row.ID', how='outer')
        if not mold.empty: df = pd.merge(df, mold, on='row.ID', how='left')
        if not derep.empty: df = pd.merge(df, derep, on='row.ID', how='left')
        if not sirius.empty: df = pd.merge(df, sirius, on='row.ID', how='left')

        # --- C. SAFEGUARDS & CLEANING (Crucial for np.select stability) ---
        required_cols = [
            'MQScore', 'FDR_mold', 'ConfidenceScoreExact_sirius', 'Tanimoto_molD_Sirius', 
            'row.m.z', 'FDR_derep', 'NPC#pathway_sirius', 'Compound_Name', 'Name_mold', 
            'name_sirius', 'Name_derep', 'Smiles', 'SMILES_mold', 'smiles_sirius', 
            'SMILES_derep', 'Adduct', 'Adduct_mold', 'adduct.y_sirius', 'Adduct_derep', 
            'adduct.x_sirius', 'molecularFormula.y_sirius', 'molecularFormula.x_sirius'
        ]
        for col in required_cols:
            if col not in df.columns: df[col] = np.nan

        for col in ['MQScore', 'FDR_mold', 'ConfidenceScoreExact_sirius', 'row.m.z', 'FDR_derep']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- D. TANIMOTO & TIERED LOGIC ---
        st.write("🧪 Calculating chemical similarities & applying rules...")
        mask_can_calc = df['SMILES_mold'].notna() & df['smiles_sirius'].notna()
        if mask_can_calc.any():
             df.loc[mask_can_calc, 'Tanimoto_molD_Sirius'] = df[mask_can_calc].apply(lambda x: calculate_tanimoto(x['SMILES_mold'], x['smiles_sirius']), axis=1)

        c_gnps = (df['MQScore'] > 0)
        c_consensus = (df['FDR_mold'] < 1) & (df['ConfidenceScoreExact_sirius'] > 0.6) & (df['Tanimoto_molD_Sirius'] > 0.85)
        c_mold_high = (df['row.m.z'] > 600) & (df['FDR_mold'] < 1)
        c_sirius_high = (df['row.m.z'] <= 600) & (df['ConfidenceScoreExact_sirius'] > 0.6)
        c_derep = (df['FDR_derep'] <= 5)
        c_mold_low = (df['FDR_mold'] < 5)
        c_sirius_low = (df['ConfidenceScoreExact_sirius'] > 0.1)
        c_canopus = df['NPC#pathway_sirius'].notna()

        conditions = [c_gnps, c_consensus, c_mold_high, c_sirius_high, c_derep, c_mold_low, c_sirius_low, c_canopus]
        choices_annot = ["GNPS/Spectral Match", "Consensus (molD + Sirius)", "molDiscovery (>600 Da)", "Sirius/CSI:FingerID (<=600 Da)", "Dereplicator+", "molDiscovery (fallback)", "Sirius/CSI:FingerID (fallback)", "Sirius/CANOPUS"]

        df['Annotated'] = np.select(conditions, choices_annot, default=None)
        
        df['Final_Name'] = np.select(conditions, [
            df['Compound_Name'], df['Name_mold'], df['Name_mold'], df['name_sirius'], 
            df['Name_derep'], df['Name_mold'], df['name_sirius'], "MolecularFormula_Class_predicted"
        ], default=None)

        df['Final_SMILES'] = np.select(conditions, [
            df['Smiles'], df['SMILES_mold'], df['SMILES_mold'], df['smiles_sirius'], 
            df['SMILES_derep'], df['SMILES_mold'], df['smiles_sirius'], None
        ], default=None)

        df['Final_Adduct'] = np.select(conditions, [
            df['Adduct'], df['Adduct_mold'], df['Adduct_mold'], df['adduct.y_sirius'], 
            df['Adduct_derep'], df['Adduct_mold'], df['adduct.y_sirius'], df['adduct.x_sirius']
        ], default=None)

        df['Final_Formula'] = np.select([c_sirius_high | c_sirius_low, c_canopus], [df['molecularFormula.y_sirius'], df['molecularFormula.x_sirius']], default=None)
        
        mask_need_formula = df['Final_Formula'].isna() & df['Final_SMILES'].notna()
        if mask_need_formula.any():
             df.loc[mask_need_formula, 'Final_Formula'] = df[mask_need_formula]['Final_SMILES'].apply(get_formula_from_smiles)

        # --- E. API & ACCURACY ---
        df_final = df.dropna(subset=['Annotated']).copy()
        st.write(f"🌍 Fetching NPClassifier for {len(df_final)} features...")
        
        df_final['Accurate_Mass'] = df_final.apply(lambda x: get_accurate_mass(x.get('Final_Formula'), x.get('Final_Adduct')), axis=1)
        df_final['ppm_error'] = ((df_final['row.m.z'] - df_final['Accurate_Mass']) / df_final['Accurate_Mass']) * 1e6

        # Initialize NPC columns with SIRIUS data if available
        for col, sirius_col in [('NPC_Pathway', 'NPC#pathway_sirius'), ('NPC_Superclass', 'NPC#superclass_sirius'), ('NPC_Class', 'NPC#class_sirius')]:
             df_final[col] = df_final.get(sirius_col, np.nan)

        # Query API for remaining SMILES
        mask_api = df_final['Final_SMILES'].notna() & df_final['NPC_Pathway'].isna()
        rows_to_query = df_final[mask_api]
        if not rows_to_query.empty:
            prog = st.progress(0)
            for i, (idx, row) in enumerate(rows_to_query.iterrows()):
                p, s, c_cls = get_npclassifier_data(row['Final_SMILES'])
                df_final.at[idx, 'NPC_Pathway'] = p
                df_final.at[idx, 'NPC_Superclass'] = s
                df_final.at[idx, 'NPC_Class'] = c_cls
                if i % 5 == 0: prog.progress(min(1.0, (i + 1) / len(rows_to_query)))
            prog.empty()

        status.update(label="✅ Merging Complete!", state="complete", expanded=False)

    # --- F. RESULTS ---
    st.success(f"🎉 Success! {len(df_final)} features annotated out of {len(df)} total.")
    
    final_cols = ['row.ID', 'row.m.z', 'row.retention.time', 'Annotated', 'Final_Name', 'Final_Formula', 'Final_Adduct', 'ppm_error', 'NPC_Pathway', 'NPC_Superclass', 'NPC_Class', 'Final_SMILES']
    existing_cols = [c for c in final_cols if c in df_final.columns]
    st.dataframe(df_final[existing_cols].head(50), use_container_width=True)

    st.download_button("⬇️ Download Annotation Table (CSV)", df_final.to_csv(index=False).encode('utf-8'), "Merged_Annotations.csv", "text/csv", type='primary')
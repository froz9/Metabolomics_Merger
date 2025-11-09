import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs
import requests
import molmass
import re

# ===================================================================
# 1. CONFIG & HELPER FUNCTIONS
# ===================================================================
st.set_page_config(page_title="Metabolomics Tiered Merger", layout="wide", page_icon="🧬")

def get_formula_from_smiles(smiles):
    try:
        if pd.isna(smiles) or smiles == "": return None
        mol = Chem.MolFromSmiles(smiles)
        if mol: return rdMolDescriptors.CalcMolFormula(mol)
        return None
    except: return None

def get_inchikey_from_smiles(smiles):
    try:
        if pd.isna(smiles) or smiles == "": return None
        mol = Chem.MolFromSmiles(smiles)
        if mol: return Chem.MolToInchiKey(mol)
        return None
    except: return None

def calculate_tanimoto(smiles1, smiles2):
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

def get_accurate_mass_robust(formula, adduct):
    """
    Calculates accurate m/z using a standardized adduct dictionary.
    Handles inputs with/without brackets and explicit charges.
    """
    if pd.isna(formula) or pd.isna(adduct) or formula == "": return np.nan
    try:
        M = molmass.Formula(formula).isotope.mass
        
        # Constants
        proton = 1.007276466
        na = 22.98976928
        k = 38.96370668
        nh4 = 18.03858
        
        # Standardized Adduct Map: (Multiplier, Mass Addition, Charge State)
        # Handles both 'M+H' and '[M+H]+' styles by normalizing input first if needed,
        # or just exhaustive listing for safety.
        adduct_rules = {
            # --- POSITIVE MODE ---
            "M+H": (1, proton, 1),
            "[M+H]+": (1, proton, 1),
            "M+Na": (1, na, 1),
            "[M+Na]+": (1, na, 1),
            "M+K": (1, k, 1),
            "[M+K]+": (1, k, 1),
            "M+NH4": (1, nh4, 1),
            "[M+NH4]+": (1, nh4, 1),
            "M+2H": (1, 2 * proton, 2),
            "[M+2H]2+": (1, 2 * proton, 2),
            "M+H-H2O": (1, proton - 18.010565, 1),
            "[M+H-H2O]+": (1, proton - 18.010565, 1),
            "M-H2O+H": (1, proton - 18.010565, 1), # Common variant
            "2M+H": (2, proton, 1),
            "[2M+H]+": (2, proton, 1),
            "2M+Na": (2, na, 1),
            "[2M+Na]+": (2, na, 1),
            
            # --- NEGATIVE MODE ---
            "M-H": (1, -proton, 1),
            "[M-H]-": (1, -proton, 1),
            "M+Cl": (1, 34.968853, 1),
            "[M+Cl]-": (1, 34.968853, 1),
            "M+FA-H": (1, 44.998201, 1), # Formic Acid
            "[M+FA-H]-": (1, 44.998201, 1),
            "2M-H": (2, -proton, 1),
            "[2M-H]-": (2, -proton, 1),
        }

        # Clean input to match keys if slight variations exist
        adduct_clean = str(adduct).strip()
        
        if adduct_clean in adduct_rules:
            mult, add, charge = adduct_rules[adduct_clean]
            return (mult * M + add) / abs(charge)
        
        # Fallback: try stripping brackets if exact match failed
        norm = adduct_clean.replace("[", "").replace("]", "")
        if norm.endswith("+") or norm.endswith("-"): norm = norm[:-1]
        
        if norm in adduct_rules:
             mult, add, charge = adduct_rules[norm]
             return (mult * M + add) / abs(charge)
             
        return np.nan
            
    except: return np.nan

def get_halogen_boron(formula):
    if pd.isna(formula): return np.nan
    # Uses regex to ensure we don't match 'Br' when looking for 'B'
    has_b = bool(re.search(r'B(?![a-z])', formula)) 
    has_halo = bool(re.search(r'(F|Cl|Br|I|At)(?![a-z])', formula))
    if has_b and has_halo: return "Both"
    if has_b: return "Boron"
    if has_halo: return "Halogen"
    return np.nan

def get_npclassifier_data(smiles):
    if pd.isna(smiles) or smiles == "": return np.nan, np.nan, np.nan
    try:
        url = f"https://npclassifier.gnps2.org/classify?smiles={requests.utils.quote(smiles)}"
        response = requests.get(url, timeout=2)
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
st.markdown("Upload outputs from FBMN, MolDiscovery, Dereplicator+, and SIRIUS.")

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
        st.error("❌ FBMN and MS1 files are mandatory.")
        st.stop()

    with st.status("🔄 Processing...", expanded=True) as status:
        # --- LOAD ---
        st.write("📝 Loading data...")
        ms1 = pd.read_csv(ms1_file)
        ms1 = ms1.rename(columns={'cluster index': 'row.ID', 'precursor mass': 'row.m.z', 'RTMean': 'row.retention.time'})
        ms1['row.ID'] = ms1['row.ID'].astype(str)
        ms1 = ms1[['row.ID', 'row.m.z', 'row.retention.time']]

        fbmn = pd.read_csv(fbmn_file, sep='\t')
        if '#Scan#' in fbmn.columns: fbmn = fbmn.rename(columns={'#Scan#': 'row.ID'})
        elif 'Scan' in fbmn.columns: fbmn = fbmn.rename(columns={'Scan': 'row.ID'})
        fbmn['row.ID'] = fbmn['row.ID'].astype(str)
        
        # Normalize common FBMN adducts to standard format
        adduct_map = {"M+H":"[M+H]+", "M+2H":"[M+2H]2+", "M+Na":"[M+Na]+", "M+NH4":"[M+NH4]+", "M-H2O+H":"[M+H-H2O]+", "M+K":"[M+K]+"}
        if 'Adduct' in fbmn.columns: fbmn['Adduct'] = fbmn['Adduct'].replace(adduct_map)

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
        else: sirius = pd.DataFrame(columns=['row.ID'])

        # --- MERGE ---
        st.write("🔗 Merging tables...")
        df = pd.merge(fbmn, ms1, on='row.ID', how='outer')
        if not mold.empty: df = pd.merge(df, mold, on='row.ID', how='left')
        if not derep.empty: df = pd.merge(df, derep, on='row.ID', how='left')
        if not sirius.empty: df = pd.merge(df, sirius, on='row.ID', how='left')

        # --- SAFEGUARDS ---
        req_cols = ['MQScore', 'FDR_mold', 'Score_mold', 'ConfidenceScoreExact_sirius', 'row.m.z', 'FDR_derep', 'Score_derep', 'NPC#pathway_sirius', 'Compound_Name', 'Name_mold', 'name_sirius', 'Name_derep', 'Smiles', 'SMILES_mold', 'smiles_sirius', 'SMILES_derep', 'Adduct', 'Adduct_mold', 'adduct.y_sirius', 'Adduct_derep', 'adduct.x_sirius', 'molecularFormula.y_sirius', 'molecularFormula.x_sirius', 'InChIKey', 'InChIkey2D_sirius']
        for col in req_cols:
            if col not in df.columns: df[col] = np.nan
        for col in ['MQScore', 'FDR_mold', 'ConfidenceScoreExact_sirius', 'row.m.z', 'FDR_derep']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- TIERED LOGIC ---
        st.write("🧪 Applying annotation rules...")
        df['Tanimoto_molD_Sirius'] = np.nan
        mask_calc = df['SMILES_mold'].notna() & df['smiles_sirius'].notna()
        if mask_calc.any():
             df.loc[mask_calc, 'Tanimoto_molD_Sirius'] = df[mask_calc].apply(lambda x: calculate_tanimoto(x['SMILES_mold'], x['smiles_sirius']), axis=1)

        conds = [
            (df['MQScore'] > 0),
            (df['FDR_mold'] < 1) & (df['ConfidenceScoreExact_sirius'] > 0.6) & (df['Tanimoto_molD_Sirius'] > 0.85),
            (df['row.m.z'] > 600) & (df['FDR_mold'] < 1),
            (df['row.m.z'] <= 600) & (df['ConfidenceScoreExact_sirius'] > 0.6),
            (df['FDR_derep'] <= 5),
            (df['FDR_mold'] < 5),
            (df['ConfidenceScoreExact_sirius'] > 0.1),
            df['NPC#pathway_sirius'].notna()
        ]
        
        df['Annotated'] = np.select(conds, [
            "GNPS/Spectral Match", "Consensus (molD + Sirius)", "molDiscovery (>600 Da)",
            "Sirius/CSI:FingerID (<=600 Da)", "Dereplicator+", "molDiscovery (fallback)",
            "Sirius/CSI:FingerID (fallback)", "Sirius/CANOPUS"
        ], default=None)

        df['Score_value'] = np.select(conds, [
            "Spectral_Match",
            df['Score_mold'].astype(str) + " (FDR:" + df['FDR_mold'].astype(str) + ")",
            df['Score_mold'].astype(str) + " (FDR:" + df['FDR_mold'].astype(str) + ")",
            df['ConfidenceScoreExact_sirius'].astype(str),
            df['Score_derep'].astype(str) + " (FDR:" + df['FDR_derep'].astype(str) + ")",
            df['Score_mold'].astype(str) + " (FDR:" + df['FDR_mold'].astype(str) + ")",
            df['ConfidenceScoreExact_sirius'].astype(str),
            "Class/Formula Predicted"
        ], default=np.nan)

        df['MSI_level'] = np.select(conds, ["2", "3", "3", "3", "3", "3", "3", "Chemical class predicted & Molecular formula predicted"], default=np.nan)
        df['Final_Name'] = np.select(conds, [df['Compound_Name'], df['Name_mold'], df['Name_mold'], df['name_sirius'], df['Name_derep'], df['Name_mold'], df['name_sirius'], "MolecularFormula_Class_predicted"], default=None)
        df['Final_SMILES'] = np.select(conds, [df['Smiles'], df['SMILES_mold'], df['SMILES_mold'], df['smiles_sirius'], df['SMILES_derep'], df['SMILES_mold'], df['smiles_sirius'], None], default=None)
        df['Final_Adduct'] = np.select(conds, [df['Adduct'], df['Adduct_mold'], df['Adduct_mold'], df['adduct.y_sirius'], df['Adduct_derep'], df['Adduct_mold'], df['adduct.y_sirius'], df['adduct.x_sirius']], default=None)
        df['Final_InChIKey'] = np.select(conds, [df['InChIKey'], None, None, df['InChIkey2D_sirius'], None, None, df['InChIkey2D_sirius'], None], default=None)
        df['Final_Formula'] = np.select([conds[3] | conds[6], conds[7]], [df['molecularFormula.y_sirius'], df['molecularFormula.x_sirius']], default=None)
        
        mask_form = df['Final_Formula'].isna() & df['Final_SMILES'].notna()
        if mask_form.any(): df.loc[mask_form, 'Final_Formula'] = df[mask_form]['Final_SMILES'].apply(get_formula_from_smiles)

        # --- POST-PROCESSING ---
        df_final = df.dropna(subset=['Annotated']).copy()
        st.write(f"🌍 Fetching NPClassifier & calculating mass for {len(df_final)} features...")
        
        df_final['Accurate.mass'] = df_final.apply(lambda x: get_accurate_mass_robust(x.get('Final_Formula'), x.get('Final_Adduct')), axis=1)
        df_final['accuracy'] = ((df_final['row.m.z'] - df_final['Accurate.mass']) / df_final['Accurate.mass']) * 1e6
        df_final['halogen_boron'] = df_final['Final_Formula'].apply(get_halogen_boron)

        mask_ikey = df_final['Final_InChIKey'].isna() & df_final['Final_SMILES'].notna()
        if mask_ikey.any(): df_final.loc[mask_ikey, 'Final_InChIKey'] = df_final[mask_ikey]['Final_SMILES'].apply(get_inchikey_from_smiles)

        for c_py, c_sir in [('NPCPathway','NPC#pathway_sirius'),('NPCSuperclass','NPC#superclass_sirius'),('NPCClass','NPC#class_sirius')]:
             df_final[c_py] = df_final.get(c_sir, np.nan)
        
        mask_api = df_final['Final_SMILES'].notna() & df_final['NPCPathway'].isna()
        rows = df_final[mask_api]
        if not rows.empty:
            prog = st.progress(0)
            for i, (idx, r) in enumerate(rows.iterrows()):
                p, s, c = get_npclassifier_data(r['Final_SMILES'])
                df_final.at[idx, 'NPCPathway'] = p
                df_final.at[idx, 'NPCSuperclass'] = s
                df_final.at[idx, 'NPCClass'] = c
                if i % 10 == 0: prog.progress(min(1.0, (i + 1) / len(rows)))
            prog.empty()

        status.update(label="✅ Complete!", state="complete", expanded=False)

    # --- FINAL EXPORT ---
    st.success(f"🎉 Annotated {len(df_final)} features!")
    r_columns = ['row.ID', 'Final_Name', 'NPCPathway', 'NPCSuperclass', 'NPCClass', 'Annotated', 'Score_value', 'Final_Formula', 'Final_Adduct', 'row.m.z', 'Accurate.mass', 'accuracy', 'row.retention.time', 'MSI_level', 'halogen_boron', 'Final_SMILES', 'Final_InChIKey']
    final_df = df_final[[c for c in r_columns if c in df_final.columns]]
    st.dataframe(final_df.head(50), use_container_width=True)
    st.download_button("⬇️ Download Final Annotation Table", final_df.to_csv(index=False).encode('utf-8'), "FinalAnnotationTable.csv", "text/csv", type='primary')
import os
import re
import time
import json
import gspread
import torch
import numpy as np
import pandas as pd
import datetime as _dt
from pathlib import Path
from tqdm.auto import tqdm
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator 

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
pd.set_option('display.max_colwidth', 240)
tqdm.pandas() # Aktivasi progress bar untuk pandas apply

# Set Base Path ke direktori tempat script ini dijalankan
BASE_PATH = Path(".").resolve()

# Konfigurasi Google Sheets
SPREADSHEET_ID = "1-qmvFshzR9bp0sGv_wXGVcj_KqsIa5dKc2W-nR92HUY"
SHEET_NAME = "Jan"

# Direktori Database & Output
CREDENTIALS_FILE = BASE_PATH / "Database" / "credentials.json"
MAINTENANCE_FILE = BASE_PATH / "Database" / "Defect Report BATMIS 1 Jul 2025 - 23 March 2026.xlsx"

OUTPUT_DIR = BASE_PATH / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "vector_matches_hybrid.json"
TOP_N = 3
SEAT_BONUS_MULT = 0.3
SIM_THRESHOLD = 0.40

TARGETS = {
    'defect_id':    "Report Number",
    'defect_date':  "Occurrence Date",
    'defect_desc':  "Report Description/ Report Title",
    'defect_ac':    "UNIT",
    'repair_id':     "DEFECT",
    'repair_date':   "RESOLVED_DATE",
    'repair_desc':   "DEFECT_DESCRIPTION",
    'repair_resol':  "RESOLUTION_DESCRIPTION",
    'repair_ata':    "CHAPTER",
    'repair_ac':     "AC"
}

# Regex 1-6
RE_SEAT_STANDARD = re.compile(r'\b([1-9][0-9]?)\s*(Row|Baris)?\s*([A-F])\b', re.IGNORECASE)
RE_SEAT_COMPACT = re.compile(r'\b([1-9][0-9]?)\s*(Row|Baris)?\s*([A-F]{2,})\b', re.IGNORECASE)
RE_DOOR_LOC = re.compile(r'\b([14])\s*([LR])\b|\b([LR])\s*([14])\b', re.IGNORECASE)
RE_FA_SEAT = re.compile(r'\bFA\s*0?([1-5])\b', re.IGNORECASE)
RE_LAVATORY = re.compile(r'\bLAV(?:ATORY)?\s*-?\s*([A-F])\b', re.IGNORECASE)
RE_GALLEY = re.compile(r'\b(?:G|GALLEY)\s*-?\s*([1-8])\b', re.IGNORECASE)

def parse_all_seats(text):
    if not text or pd.isna(text): return set(), set()
    txt_upper = str(text).upper()
    seats, rows = set(), set()

    for row_num, _, seat_let in RE_SEAT_STANDARD.findall(txt_upper):
        row_id = str(int(row_num))
        rows.add(row_id); seats.add(f"{row_id}{seat_let}")

    for row_num, _, seat_letters_str in RE_SEAT_COMPACT.findall(txt_upper):
        row_id = str(int(row_num))
        rows.add(row_id)
        for char in seat_letters_str:
            if char in 'ABCDEF': seats.add(f"{row_id}{char}")

    for m in RE_DOOR_LOC.findall(txt_upper):
        num, letter = m[0] or m[3], m[1] or m[2]
        seats.add(f"{num}{letter}")

    for fa_num in RE_FA_SEAT.findall(txt_upper): seats.add(f"FA{int(fa_num)}")
    for lav_letter in RE_LAVATORY.findall(txt_upper): seats.add(f"LAV_{lav_letter}")
    for galley_num in RE_GALLEY.findall(txt_upper): seats.add(f"G{galley_num}")

    return seats, rows

def compute_seat_score(defect_text, repair_text):
    d_seats, d_rows = parse_all_seats(defect_text)
    r_seats, r_rows = parse_all_seats(repair_text)
    if not d_seats or not r_seats: return 0.0
    if d_seats.intersection(r_seats): return 1.0
    if d_rows.intersection(r_rows): return 0.5
    return 0.0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_data(file_path):
    path = Path(file_path)
    if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_excel(path, engine='openpyxl')
    print(f"Loaded Excel: {path.name}")
    return df

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_col_safe(target, cols):
    if target is None: return None
    target_low = str(target).lower().strip()
    cols_map = {str(c).lower(): c for c in cols}
    if target_low in cols_map: return cols_map[target_low]
    for c in cols:
        if target_low in str(c).lower(): return c
    return None

def excel_serial_to_date(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)): return pd.NaT
        return pd.to_datetime(_dt.datetime(1899, 12, 30) + _dt.timedelta(days=float(val)))
    except: return pd.NaT

def is_mostly_numeric_series(s, threshold=0.6):
    s_nonnull = s.dropna().astype(str).str.strip()
    if s_nonnull.empty: return False
    return s_nonnull.apply(lambda x: x.replace('.', '', 1).isdigit()).mean() >= threshold

def force_to_datetime(df, colname):
    if colname is None or colname not in df.columns: return df
    series = df[colname]
    if is_mostly_numeric_series(series):
        df[colname] = series.apply(excel_serial_to_date)
    else:
        df[colname] = pd.to_datetime(series, errors='coerce')
    df[colname] = pd.to_datetime(df[colname]).dt.tz_localize(None)
    return df

def translate_to_en(text):
    """Fungsi untuk translate teks ke bahasa Inggris menggunakan Google Translate API."""
    if not text or pd.isna(text) or str(text).strip() == "":
        return ""
    try:
        return GoogleTranslator(source='auto', target='en').translate(str(text))
    except Exception as e:
        return str(text)

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================
def main():
    print("Memulai Local Vector Search Engine...")

    # 1. LOAD DATA GOOGLE SHEET (Defects) menggunakan Service Account
    print(f"\nAuthenticating Google Sheets for: '{SHEET_NAME}'...")
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(f"Missing credentials file: {CREDENTIALS_FILE}. Pastikan file JSON service account tersedia.")
        
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)
    
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    df_defects = normalize_columns(pd.DataFrame(sheet.get_all_records()))
    print(f"Success! '{SHEET_NAME}' loaded dari Google Sheets. Shape: {df_defects.shape}")

    # 2. LOAD DATA EXCEL (Repairs)
    print("\nLoading Maintenance repairs dari local storage...")
    df_repairs = normalize_columns(load_data(MAINTENANCE_FILE))

    # 3. COLUMN MAPPING & DATETIME
    mapped = {}
    for key, target in TARGETS.items():
        src_df = df_defects if key.startswith('defect') else df_repairs
        mapped[key] = find_col_safe(target, src_df.columns)

    cols_map = {
        'DEFECT_ID_COL': mapped.get('defect_id'), 'DEFECT_DATE_COL': mapped.get('defect_date'),
        'DEFECT_DESC_COL': mapped.get('defect_desc'), 'DEFECT_AC_COL': mapped.get('defect_ac'),
        'REPAIR_ID_COL': mapped.get('repair_id'), 'REPAIR_DATE_COL': mapped.get('repair_date'),
        'REPAIR_DESC_COL': mapped.get('repair_desc'), 'REPAIR_RESOL_COL': mapped.get('repair_resol'),
        'REPAIR_AC_COL': mapped.get('repair_ac')
    }

    df_defects = force_to_datetime(df_defects, cols_map['DEFECT_DATE_COL'])
    df_repairs = force_to_datetime(df_repairs, cols_map['REPAIR_DATE_COL'])

    # 4. INIT AI MODEL
    print("\nLoading Sentence Transformer Model (all-MiniLM-L6-v2) for English...")
    # Smart device detection: CUDA (Nvidia), MPS (Apple Silicon), or CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Menggunakan hardware device: {device.upper()}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # 5. FASE TRANSLATION
    print("\n[FASE TRANSLATION] Men-translate Data Defect ke Bahasa Inggris...")
    start_trans = time.time()

    print("Translating Defects... (Mohon tunggu sebentar)")
    df_defects['DEFECT_DESC_EN'] = df_defects[cols_map['DEFECT_DESC_COL']].astype(str).progress_apply(translate_to_en)

    print(f"Translasi selesai dalam {time.time() - start_trans:.2f} detik.")

    # 6. FASE INGESTION
    print("\n[FASE INGESTION] Mengubah Database Repair menjadi Vektor (Teks Asli)...")
    start_ingest = time.time()

    df_repairs['COMBINED_TEXT'] = df_repairs[cols_map['REPAIR_DESC_COL']].astype(str).fillna("") + " " + df_repairs[cols_map['REPAIR_RESOL_COL']].astype(str).fillna("")
    repair_texts = df_repairs['COMBINED_TEXT'].tolist()

    repair_embeddings = model.encode(
        repair_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32
    )
    print(f"Ingestion selesai dalam {time.time() - start_ingest:.2f} detik.")

    # 7. FASE RETRIEVAL / MATCHING
    out = []
    total_defects = len(df_defects)
    start_matching = time.time()

    print(f"\n[FASE RETRIEVAL] Matching {total_defects} defects...")

    for idx, defect in tqdm(df_defects.iterrows(), total=total_defects, desc="Vector Matching"):
        defect_identifier = defect.get(cols_map['DEFECT_ID_COL'], f"row_{idx}")
        defect_desc_original = str(defect.get(cols_map['DEFECT_DESC_COL'], "")).strip()
        defect_desc_en = str(defect.get('DEFECT_DESC_EN', "")).strip()
        defect_date = defect.get(cols_map['DEFECT_DATE_COL'])
        defect_reg  = str(defect.get(cols_map['DEFECT_AC_COL'], "")).strip().upper()

        mask_ac = df_repairs[cols_map['REPAIR_AC_COL']].astype(str).str.upper() == defect_reg
        mask_date = True
        if pd.notna(defect_date) and cols_map['REPAIR_DATE_COL'] in df_repairs.columns:
            mask_date = df_repairs[cols_map['REPAIR_DATE_COL']] >= defect_date

        candidate_indices = df_repairs[mask_ac & mask_date].index.tolist()

        if not candidate_indices:
            out.append({
                "defect_identifier": defect_identifier, "defect_date": defect_date,
                "defect_reg": defect_reg,
                "defect_desc": defect_desc_original,
                "defect_desc_en": defect_desc_en,
                "matched": False
            })
            continue

        defect_vector = model.encode(defect_desc_en, convert_to_tensor=True)

        candidate_vectors = repair_embeddings[candidate_indices]
        cos_scores = util.cos_sim(defect_vector, candidate_vectors)[0].cpu().numpy()

        results = []
        for i, original_idx in enumerate(candidate_indices):
            semantic_score = float(cos_scores[i])
            if semantic_score < SIM_THRESHOLD: continue

            r = df_repairs.iloc[original_idx]
            rep_text_original = r['COMBINED_TEXT']

            seat_bonus = compute_seat_score(defect_desc_original, rep_text_original) * SEAT_BONUS_MULT
            final_score = min(semantic_score + seat_bonus, 2)

            results.append({
                "repair_id": r.get(cols_map['REPAIR_ID_COL']),
                "repair_date": r.get(cols_map['REPAIR_DATE_COL']),
                "repair_ac": r.get(cols_map['REPAIR_AC_COL']),
                "repair_desc": str(r.get(cols_map['REPAIR_DESC_COL'])),
                "repair_resolution": str(r.get(cols_map['REPAIR_RESOL_COL'])),
                "semantic_score": round(semantic_score, 4),
                "seat_bonus": round(seat_bonus, 4),
                "final_score": round(final_score, 4)
            })

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)[:TOP_N]

        if results:
            for rank, m in enumerate(results, start=1):
                out.append({
                    "defect_identifier": defect_identifier, "defect_date": defect_date,
                    "defect_reg": defect_reg,
                    "defect_desc": defect_desc_original,
                    "defect_desc_en": defect_desc_en,
                    "matched": True, "match_rank": rank, **m
                })
        else:
            out.append({
                "defect_identifier": defect_identifier, "defect_date": defect_date,
                "defect_reg": defect_reg,
                "defect_desc": defect_desc_original,
                "defect_desc_en": defect_desc_en,
                "matched": False
            })

    # 8. SIMPAN KE JSON
    print(f"\n--- MATCHING SELESAI ---")
    print(f"Total waktu matching (Retrieval): {time.time() - start_matching:.2f} detik.")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print(f"Hasil disimpan di: {OUT_FILE}")

if __name__ == "__main__":
    main()

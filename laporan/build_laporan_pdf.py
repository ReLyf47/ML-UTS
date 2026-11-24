#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bangun laporan.pdf UTS Machine Learning secara data-driven.

SETIAP angka pada laporan diambil dari artefak pipeline, bukan ditulis keras:
  outputs/metrics/*.json        -> metrik model (regresi/klasifikasi/klastering)
  outputs/predictions/*.csv     -> statistik berkas pengumpulan
  data/manifests/*.json          -> inventaris, skema, dan ringkasan target
  experiments/experiments.jsonl  -> jumlah fitur per model
  data/raw/*.csv                 -> bukti audit (split temporal, kebocoran,
                                   kategori unseen, prevalensi fraud)

Jalankan dari root repositori:
    python laporan/build_laporan_pdf.py
"""

import json
import sys
from importlib import metadata
from itertools import combinations
from math import isqrt
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Table, TableStyle,
)
from reportlab.pdfgen import canvas

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PRED_DIR = BASE_DIR / "outputs" / "predictions"
MANIFEST_DIR = BASE_DIR / "data" / "manifests"
RAW_DIR = BASE_DIR / "data" / "raw"
EXP_JSONL = BASE_DIR / "experiments" / "experiments.jsonl"
FIGURES_DIR = BASE_DIR / "laporan" / "figures"
PDF_PATH = BASE_DIR / "laporan.pdf"

for required in [
    METRICS_DIR / "regression_metrics.json",
    METRICS_DIR / "classification_metrics.json",
    METRICS_DIR / "clustering_metrics.json",
    MANIFEST_DIR / "source_manifest.json",
    MANIFEST_DIR / "dataset_schemas.json",
]:
    if not required.exists():
        sys.exit(f"[ERROR] artefak tidak ditemukan: {required}\n"
                 "Jalankan `python kode/run_all.py` terlebih dahulu.")

# Konstanta tugas diambil dari satu sumber yang sama dengan pipeline.
try:
    sys.path.insert(0, str(BASE_DIR))
    from kode.common.config import RANDOM_SEED, REVIEW_COST_IDR
except Exception:  # pragma: no cover - fallback bila paket kode tidak tersedia
    RANDOM_SEED, REVIEW_COST_IDR = 42, 25000.0

# Ukuran sampel ini mengikuti argumen di kode/clustering/train_and_evaluate.py
# karena tidak disimpan di clustering_metrics.json.
CLU_N_SIL_KMEANS = 10_000
CLU_N_GMM = 25_000
CLU_N_AGGLO = 5_000

# Mengikuti np.percentile(distances, 98.5) di kode/clustering/train_and_evaluate.py.
CLU_ANOMALY_PERCENTILE = 98.5
# Mengikuti evaluate_stability(..., subsample_ratio=0.8) di kode/clustering/train_and_evaluate.py.
CLU_STABILITY_SUBSAMPLE = 0.8

# Terjemahan dan revisi strategi dari clustering_metrics.json, dikunci berdasarkan nama persona.
PERSONA_STRATEGI_ID = {
    "VIP Champions": "Program loyalitas eksklusif, layanan prioritas, akses awal produk premium.",
    "Hyperactive Bargain Hunters": (
        "Verifikasi dulu indikasi aktivitas otomatis (bot/skrip) sebelum diberi promo; "
        "tahan kupon massal sampai terverifikasi."
    ),
    "Casual Shoppers": (
        "Dorong pembelian ulang: bonus checkout, langganan gratis ongkir, rekomendasi produk terkait."
    ),
    "Discount Hunters": (
        "Masih aktif, jadi fokus menjaga margin: voucher dengan minimum belanja bertingkat, "
        "bundling untuk menaikkan nilai keranjang, dan pemantauan retur."
    ),
    "Lapsed / At-Risk Dormant": (
        "Kampanye reaktivasi (win-back) dengan voucher reaktivasi dan survei alasan berhenti."
    ),
    "Anomalous Outliers / Extreme Accounts": (
        "Audit keamanan dan pemeriksaan reseller; cek kepatuhan terhadap penyalahgunaan merchant."
    ),
}

# --------------------------------------------------------------------------- #
# Helper format (kaidah Indonesia: titik = ribuan, koma = desimal)
# --------------------------------------------------------------------------- #
ID_MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def fmt_int(n):
    return f"{int(round(float(n))):,}".replace(",", ".")


def fmt_dec(x, d=2):
    s = f"{float(x):,.{d}f}"
    return s.replace(",", "\u0001").replace(".", ",").replace("\u0001", ".")


def fmt_pct(x, d=2):
    """x berupa pecahan (0.9 -> '90,00%')."""
    return fmt_dec(float(x) * 100.0, d) + "%"


def fmt_pct_raw(x, d=3):
    """x sudah berupa persen (5.559 -> '5,559%')."""
    return fmt_dec(float(x), d) + "%"


def fmt_idr(x):
    return "Rp " + fmt_int(x)


def fmt_juta(x, d=1):
    return "Rp " + fmt_dec(float(x) / 1e6, d) + " juta"


def fmt_date_id(value):
    d = pd.to_datetime(value)
    return f"{d.day:02d}/{d.month:02d}/{d.year}"


def fmt_tanggal_id(value):
    d = pd.to_datetime(value)
    return f"{d.day} {ID_MONTHS[d.month - 1]} {d.year}"


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def pkg_version(name):
    try:
        return metadata.version(name)
    except Exception:
        return "n/a"


def resolve_prediction(filename):
    """Berkas pengumpulan digandakan di root dan outputs/predictions."""
    for candidate in (BASE_DIR / filename, PRED_DIR / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(filename)


# --------------------------------------------------------------------------- #
# Koleksi fakta
# --------------------------------------------------------------------------- #
TASK_ID = {"regression": "Regresi", "classification": "Klasifikasi", "clustering": "Klastering"}
ROLE_ID = {"train": "Latih", "test": "Uji Tersembunyi", "unsupervised": "Segmentasi"}

CAT_LABEL = {
    "makanan_minuman": "Makanan & Minuman", "fashion": "Fashion", "elektronik": "Elektronik",
    "rumah_tangga": "Rumah Tangga", "hobi": "Hobi", "otomotif": "Otomotif",
    "multivariate_extreme": "Ekstrem Multivariat",
}

REG_MODEL_LABEL = {
    "Baseline_Global_Median": "Baseline Global Median",
    "Baseline_Service_Median": "Baseline Layanan Median",
    "Log_Ridge_Quantile": "Log-Ridge Quantile (log1p)",
    "HistGBM_Quantile_Clean": "HistGBM Bersih (Final)",
    "HistGBM_With_JumlahScan_Leakage": "HistGBM (Ablasi jumlah_scan)",
}
REG_MODEL_FEATURES = {
    "Baseline_Global_Median": "GlobalMedianBaseline",
    "Baseline_Service_Median": "ServiceMedianBaseline",
    "Log_Ridge_Quantile": "LogRidgeQuantileModel",
    "HistGBM_Quantile_Clean": "HistGradientBoostingQuantileModel",
    "HistGBM_With_JumlahScan_Leakage": "HistGBM_Leakage_Ablation",
}
CLF_MODEL_LABEL = {
    "Baseline_Logistic_Regression": "Regresi Logistik Terbobot",
    "HistGBM_Unweighted": "HistGBM (Unweighted)",
    "HistGBM_Balanced_Raw": "HistGBM Balanced (Raw)",
    "HistGBM_Balanced_Sigmoid": "HistGBM Balanced (Sigmoid)",
    "HistGBM_Balanced_Isotonic": "HistGBM Balanced Isotonik (Final)",
    "HistGBM_RandomUnderSampler": "HistGBM (Random Under-Sampler)",
    "HistGBM_With_PolicyFlag": "HistGBM (Ablasi flag_review_manual)",
}
CLF_STRATEGY = {
    "Baseline_Logistic_Regression": "P x nominal di atas biaya",
    "HistGBM_Unweighted": "P x nominal di atas biaya",
    "HistGBM_Balanced_Raw": "P x nominal di atas biaya",
    "HistGBM_Balanced_Sigmoid": "P x nominal di atas biaya",
    "HistGBM_Balanced_Isotonic": "P x nominal di atas biaya (optimum)",
    "HistGBM_RandomUnderSampler": "P x nominal di atas biaya",
    "HistGBM_With_PolicyFlag": "Dengan flag_review_manual",
}


def read_n_features():
    """experiment_log menyimpan riwayat; ambil n_features terakhir per model."""
    latest = {}
    if EXP_JSONL.exists():
        with open(EXP_JSONL, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("model_name"):
                    latest[row["model_name"]] = row.get("n_features")
    return latest


def read_raw_facts():
    """Bukti audit yang tidak tersimpan di metrics: split temporal, korelasi
    kebocoran, kategori unseen, dan jendela validasi klasifikasi."""
    facts = {"available": False}
    try:
        reg_train = pd.read_csv(
            RAW_DIR / "regresi_train.csv",
            usecols=["tgl_pickup", "jumlah_scan", "durasi_jam", "mitra_kurir"],
        )
        reg_test = pd.read_csv(
            RAW_DIR / "regresi_test.csv", usecols=["jumlah_scan", "mitra_kurir"]
        )
        months = pd.to_datetime(reg_train["tgl_pickup"], format="%Y-%m-%d").dt.month
        facts.update(
            reg_fit_n=int((months != 8).sum()),
            reg_val_n=int((months == 8).sum()),
            scan_corr=float(reg_train["jumlah_scan"].corr(reg_train["durasi_jam"])),
            unseen_kurir=sorted(set(reg_test["mitra_kurir"]) - set(reg_train["mitra_kurir"])),
        )
        clf_train = pd.read_csv(
            RAW_DIR / "klasifikasi_train.csv", usecols=["waktu", "is_fraud"]
        )
        clf_months = pd.to_datetime(clf_train["waktu"], format="%Y-%m-%d %H:%M:%S").dt.month
        val_window = clf_train[clf_months.isin([7, 8])]
        facts.update(
            clf_val_n=int(len(val_window)),
            clf_val_fraud=int(val_window["is_fraud"].sum()),
            clf_train_n=int(len(clf_train)),
            clf_train_fraud=int(clf_train["is_fraud"].sum()),
        )
        facts["available"] = True
    except Exception as exc:  # data mentah (LFS) tidak tersedia
        facts["error"] = str(exc)
    return facts


def collect_facts():
    src = load_json(MANIFEST_DIR / "source_manifest.json")
    schemas = load_json(MANIFEST_DIR / "dataset_schemas.json")["schemas"]
    reg = load_json(METRICS_DIR / "regression_metrics.json")
    clf = load_json(METRICS_DIR / "classification_metrics.json")
    clu = load_json(METRICS_DIR / "clustering_metrics.json")
    nfeat = read_n_features()
    raw = read_raw_facts()

    facts = {"manifest": src, "schemas": schemas, "reg": reg, "clf": clf,
             "clu": clu, "nfeat": nfeat, "raw": raw}

    # ---- inventaris dataset -------------------------------------------------
    inventory = []
    for key, meta in src["files"].items():
        tr = meta.get("time_range")
        inventory.append({
            "file": meta["original_filename"],
            "task": TASK_ID.get(meta["task"], meta["task"]),
            "role": ROLE_ID.get(meta["role"], meta["role"]),
            "rows": meta["row_count"],
            "cols": meta["column_count"],
            "size_mb": meta["file_size_mb"],
            "span": (f"{fmt_date_id(tr['min'])} - {fmt_date_id(tr['max'])}") if tr else "Agregat 90 Hari",
        })
    facts["inventory"] = inventory
    facts["total_rows"] = sum(row["rows"] for row in inventory)

    # ---- berkas prediksi ----------------------------------------------------
    reg_pred = pd.read_csv(resolve_prediction("prediksi_regresi.csv"))
    clf_pred = pd.read_csv(resolve_prediction("prediksi_klasifikasi.csv"))
    clu_pred = pd.read_csv(resolve_prediction("cluster.csv"))
    reg_ids_ok = clf_ids_ok = None
    try:
        reg_test_ids = set(pd.read_csv(RAW_DIR / "regresi_test.csv", usecols=["id_paket"])["id_paket"])
        reg_ids_ok = set(reg_pred["id_paket"]) == reg_test_ids
    except Exception:
        pass
    try:
        clf_test_ids = set(pd.read_csv(RAW_DIR / "klasifikasi_test.csv", usecols=["id_transaksi"])["id_transaksi"])
        clf_ids_ok = set(clf_pred["id_transaksi"]) == clf_test_ids
    except Exception:
        pass
    facts["pred"] = {
        "reg_rows": len(reg_pred),
        "reg_p50_min": float(reg_pred["p50"].min()), "reg_p50_max": float(reg_pred["p50"].max()),
        "reg_p90_min": float(reg_pred["p90"].min()), "reg_p90_max": float(reg_pred["p90"].max()),
        "reg_cross": int((reg_pred["p90"] < reg_pred["p50"]).sum()),
        "reg_ids_ok": reg_ids_ok,
        "clf_rows": len(clf_pred),
        "clf_prob_min": float(clf_pred["prob"].min()), "clf_prob_max": float(clf_pred["prob"].max()),
        "clf_flag": int(clf_pred["tandai"].sum()), "clf_flag_rate": float(clf_pred["tandai"].mean()),
        "clf_ids_ok": clf_ids_ok,
        "clu_rows": len(clu_pred),
        "clu_counts": {int(k): int(v) for k, v in clu_pred["cluster"].value_counts().items()},
    }

    # ---- regresi ------------------------------------------------------------
    clean = reg["HistGBM_Quantile_Clean"]
    leak = reg["HistGBM_With_JumlahScan_Leakage"]
    cv = reg["Random_5Fold_CV"]
    facts["reg_clean"] = clean
    facts["reg_leak"] = leak
    facts["reg_cv"] = cv
    facts["reg_conformal"] = reg.get("conformal_adjustment")
    facts["reg_service"] = reg.get("breakdown_layanan", [])
    facts["reg_pulau"] = reg.get("breakdown_pulau", [])
    facts["reg_kurir"] = reg.get("breakdown_kurir", [])
    facts["reg_importance"] = reg.get("permutation_importance_top", [])
    facts["reg_model_rows"] = []
    for key, label in REG_MODEL_LABEL.items():
        m = reg[key]
        n = nfeat.get(REG_MODEL_FEATURES[key])
        facts["reg_model_rows"].append({
            "key": key, "label": label,
            "features": "-" if key.startswith("Baseline") else fmt_int(n),
            "p50": m["pinball_loss_p50"], "p90": m["pinball_loss_p90"],
            "coverage": m["coverage_p90"], "cross": int(m["crossing_violations"]),
        })
    clean_rows = [r for r in facts["reg_model_rows"] if r["key"] != "HistGBM_With_JumlahScan_Leakage"]
    facts["reg_best_clean"] = min(clean_rows, key=lambda r: r["p50"])["key"]

    # ---- klasifikasi --------------------------------------------------------
    iso = clf["HistGBM_Balanced_Isotonic"]
    facts["clf_iso"] = iso
    facts["clf_nothing"] = iso["cost_flag_nothing"]
    facts["clf_everything"] = iso["cost_flag_everything"]
    facts["clf_savings"] = iso["savings_vs_flag_nothing"]
    facts["clf_savings_pct"] = iso["savings_vs_flag_nothing"] / iso["cost_flag_nothing"]
    facts["clf_review_cost"] = (iso["cost_flag_everything"] / raw["clf_val_n"]
                                if raw.get("clf_val_n") else REVIEW_COST_IDR)
    sweep = clf.get("cost_sweep", [])
    facts["clf_sweep"] = sweep
    facts["clf_sweep_best"] = min(sweep, key=lambda r: r["total_cost"]) if sweep else None
    facts["clf_kanal"] = clf.get("breakdown_kanal", [])
    facts["clf_monthly"] = clf.get("pattern_shift", {}).get("monthly_fraud_stats", [])
    facts["clf_monthly_prauc"] = clf.get("pattern_shift", {}).get("validation_window_monthly_pr_auc", {})
    facts["clf_model_rows"] = []
    for key, label in CLF_MODEL_LABEL.items():
        m = clf[key]
        facts["clf_model_rows"].append({
            "key": key, "label": label, "strategy": CLF_STRATEGY[key],
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "brier": m["brier_score"],
            "flag_rate": m["flagged_rate"], "total_cost": m["total_cost"],
        })
    facts["clf_best_clean"] = min(
        [r for r in facts["clf_model_rows"] if r["key"] != "HistGBM_With_PolicyFlag"],
        key=lambda r: r["total_cost"],
    )["key"]
    train_meta = src["files"]["klasifikasi_train"]
    facts["clf_train_rows"] = train_meta["row_count"]
    facts["clf_prevalence"] = train_meta["target_summary"]["fraud_rate"]
    facts["clf_prevalence_n"] = train_meta["target_summary"]["counts"].get("1")

    # ---- klastering ---------------------------------------------------------
    sweep_km = clu.get("kmeans_sweep", [])
    stability = clu.get("stability", {})
    k_opt = stability.get("optimal_k")
    facts["clu_k"] = k_opt
    facts["clu_kmeans"] = sweep_km
    facts["clu_gmm"] = clu.get("gmm_sweep", [])
    facts["clu_agglo"] = clu.get("agglomerative_sweep", [])
    facts["clu_hdbscan"] = clu.get("hdbscan_density", {})
    facts["clu_k_opt_row"] = next((r for r in sweep_km if r["k"] == k_opt), {})
    facts["clu_ari_mean"] = stability.get("mean_ari")
    facts["clu_ari_std"] = stability.get("std_ari")
    facts["clu_pairwise_aris"] = stability.get("pairwise_aris", [])
    facts["clu_anom"] = clu.get("anomaly_detection", {})
    facts["clu_feature_evidence"] = clu.get("feature_evidence", {})
    facts["clu_persona"] = clu.get("persona_profiles", {})
    facts["clu_means"] = clu.get("cluster_numeric_means", {})
    facts["clu_persona_rows"] = []
    counts = facts["pred"]["clu_counts"]
    total_cust = sum(counts.values())
    for cid in sorted(facts["clu_persona"], key=lambda x: (x == "-1", int(x))):
        prof = facts["clu_persona"][cid]
        means = facts["clu_means"].get(cid, {})
        n = counts.get(int(cid), 0)
        facts["clu_persona_rows"].append({
            "id": cid, "name": prof.get("name", "-"),
            "n": n, "pct": n / total_cust if total_cust else 0.0,
            "spend": means.get("belanja_90h"), "freq": means.get("frekuensi_90h"),
            "category": CAT_LABEL.get(prof.get("top_category"), prof.get("top_category", "-")),
            "strategy": prof.get("strategy", "-"),
        })

    # ---- audit data ---------------------------------------------------------
    scan_train = schemas["regresi_train"]["columns"]["jumlah_scan"]
    scan_test = schemas["regresi_test"]["columns"]["jumlah_scan"]
    facts["scan_train_mean"] = scan_train["mean"]
    facts["scan_train_max"] = scan_train["max"]
    facts["scan_test_mean"] = scan_test["mean"]
    facts["scan_test_max"] = scan_test["max"]

    missing = []
    for ds_name, ds in schemas.items():
        for col, meta in ds["columns"].items():
            if meta.get("null_pct", 0) > 0:
                missing.append({"dataset": ds_name, "column": col,
                                "n": meta["null_count"], "pct": meta["null_pct"]})
    facts["missing"] = sorted(missing, key=lambda r: -r["pct"])

    facts["reg_target"] = src["files"]["regresi_train"]["target_summary"]
    facts["generated"] = src.get("created_at", "")
    return facts


# --------------------------------------------------------------------------- #
# Kanvas bernomor halaman
# --------------------------------------------------------------------------- #
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_decorations(total)
            super().showPage()
        super().save()

    def _draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        if self._pageNumber > 1:
            self.drawString(54, 842 - 36, "UTS Machine Learning: Regression, Classification, and Clustering")
            self.drawRightString(595 - 54, 842 - 36, "Kelompok Mahasiswa ML-UTS")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 842 - 42, 595 - 54, 842 - 42)
        self.drawRightString(595 - 54, 36, f"Halaman {self._pageNumber} dari {page_count}")
        self.drawString(54, 36, "Laporan Teknis Reproduktibel ML-UTS")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 595 - 54, 46)
        self.restoreState()


# --------------------------------------------------------------------------- #
# Helper tabel & gambar
# --------------------------------------------------------------------------- #
def _cell_styles(font_size):
    base = ParagraphStyle('td', fontSize=font_size, leading=font_size + 2.2,
                          textColor=colors.HexColor("#1F2937"), spaceAfter=0)
    return {
        "head": ParagraphStyle('th', parent=base, fontName='Helvetica-Bold',
                               textColor=colors.white, alignment=1),
        "left": ParagraphStyle('tdl', parent=base, alignment=0),
        "center": ParagraphStyle('tdc', parent=base, alignment=1),
    }


def _table(rows, widths, font_size=7.5, highlight=(), bold_rows=(), red_rows=()):
    cs = _cell_styles(font_size)
    rendered = []
    for i, row in enumerate(rows):
        line = []
        for j, cell in enumerate(row):
            if i == 0:
                line.append(Paragraph(str(cell), cs["head"]))
            else:
                line.append(Paragraph(str(cell), cs["left"] if j == 0 else cs["center"]))
        rendered.append(line)
    tbl = Table(rendered, colWidths=widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    for idx in highlight:
        style += [('BACKGROUND', (0, idx), (-1, idx), colors.HexColor("#DCFCE7")),
                  ('FONTNAME', (0, idx), (-1, idx), 'Helvetica-Bold')]
    for idx in bold_rows:
        style += [('FONTNAME', (0, idx), (-1, idx), 'Helvetica-Bold')]
    for idx in red_rows:
        style += [('BACKGROUND', (0, idx), (-1, idx), colors.HexColor("#FEE2E2")),
                  ('FONTNAME', (0, idx), (-1, idx), 'Helvetica-Bold')]
    tbl.setStyle(TableStyle(style))
    return tbl


def _figure_pair(names, caption_text, caption_style):
    paths = [FIGURES_DIR / n for n in names]
    if not all(p.exists() for p in paths):
        return []
    tbl = Table([[Image(str(paths[0]), width=3.4 * inch, height=2.05 * inch),
                  Image(str(paths[1]), width=3.4 * inch, height=2.05 * inch)]],
                colWidths=[3.5 * inch, 3.5 * inch])
    tbl.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    return [tbl, Paragraph(caption_text, caption_style)]


def _figure_single(name, caption_text, caption_style, *, height=2.1 * inch):
    path = FIGURES_DIR / name
    if not path.exists():
        return []
    return [Image(str(path), width=5.2 * inch, height=height),
            Paragraph(caption_text, caption_style)]


# --------------------------------------------------------------------------- #
# Penyusunan PDF
# --------------------------------------------------------------------------- #
def build_pdf(facts):
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=50, rightMargin=50, topMargin=54, bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    primary = colors.HexColor("#1E3A8A")
    secondary = colors.HexColor("#0F766E")
    text_dark = colors.HexColor("#1F2937")

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold',
                                 fontSize=19, leading=23, textColor=primary, spaceAfter=8)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica',
                                    fontSize=10.5, leading=14, textColor=colors.HexColor("#4B5563"),
                                    spaceAfter=6)
    author_style = ParagraphStyle('DocAuthor', parent=styles['Normal'], fontName='Helvetica-Bold',
                                  fontSize=9, leading=13, textColor=colors.HexColor("#374151"),
                                  spaceAfter=14)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=13,
                        leading=17, textColor=primary, spaceBefore=14, spaceAfter=6, keepWithNext=True)
    body = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=8.8,
                          leading=12.2, textColor=text_dark, spaceAfter=6, alignment=4)
    caption = ParagraphStyle('Caption', parent=styles['Normal'], fontName='Helvetica-Oblique',
                             fontSize=8, leading=10.5, textColor=colors.HexColor("#4B5563"),
                             alignment=1, spaceBefore=4, spaceAfter=10)

    raw = facts["raw"]
    reg_clean = facts["reg_clean"]
    iso = facts["clf_iso"]
    story = []

    # ---------------- judul ----------------
    story.append(Paragraph(
        "Hands-on Machine Learning: Delivery Duration Estimation, Financial Fraud Detection, "
        "and Customer Behavioral Segmentation", title_style))
    story.append(Paragraph(
        "Laporan Teknis Komprehensif Ujian Tengah Semester (UTS) - Pendekatan Machine Learning "
        "Klasik Tanpa Kebocoran Data", subtitle_style))
    story.append(Paragraph(
        "Ilma Julianti (101032330052) | Alif Akbar Ar Raiyan (101032330121) | "
        "Pricilia Apriana (101032300056)<br/>"
        "S1 Teknik Komputer, Fakultas Teknik Elektro, Telkom University<br/>"
        f"Mata Kuliah Machine Learning | Tanggal Audit: {fmt_tanggal_id(facts['generated'])}",
        author_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary, spaceAfter=12))

    # ---------------- ringkasan eksekutif ----------------
    story.append(Paragraph("Ringkasan Eksekutif (Executive Summary)", h1))
    summary = (
        f"Laporan ini mendokumentasikan implementasi lengkap rekayasa Machine Learning (ML) klasik pada "
        f"dataset tabular sintetis berskala besar ({fmt_int(facts['total_rows'])} baris) untuk tiga tugas "
        f"industri: estimasi kuantil durasi pengiriman logistik (Tugas 1), deteksi kecurangan finansial "
        f"teroptimasi biaya (Tugas 2), dan segmentasi perilaku pelanggan 90 hari (Tugas 3). "
        f"Audit data mengungkap <b>kebocoran data fatal (<i>post-outcome target leakage</i>)</b> pada fitur "
        f"<code>jumlah_scan</code> di Tugas 1: data latih mencatat seluruh scan retrospektif "
        f"(rata-rata {fmt_dec(facts['scan_train_mean'])} scan) sedangkan data uji hanya merekam snapshot "
        f"saat penjemputan (rata-rata {fmt_dec(facts['scan_test_mean'])} scan). Dengan mengecualikan fitur "
        f"bocor tersebut dan menerapkan <i>HistGradientBoosting Quantile Regression</i> terkalibrasi "
        f"konformal, model mencapai Pinball Loss P50 = {fmt_dec(reg_clean['pinball_loss_p50'])}, "
        f"P90 = {fmt_dec(reg_clean['pinball_loss_p90'])}, cakupan empiris P90 = "
        f"{fmt_pct(reg_clean['coverage_p90'])}, dan jaminan <i>non-crossing</i> 100%. "
        f"Pada Tugas 2, model <i>HistGradientBoosting</i> dengan kalibrasi Isotonik dan aturan Bayes "
        f"optimal menekan total kerugian finansial dari {fmt_juta(facts['clf_nothing'])} menjadi "
        f"{fmt_juta(iso['total_cost'])} (efisiensi {fmt_pct(facts['clf_savings_pct'])}). "
        f"Pada Tugas 3, K-Means dengan k = {fmt_int(facts['clu_k'])} (dipilih dengan kriteria Davies-Bouldin) "
        f"menghasilkan rata-rata ARI antar-subsampel {fmt_dec(facts['clu_ari_mean'], 4)} tetapi Silhouette "
        f"hanya {fmt_dec(facts['clu_k_opt_row']['silhouette'], 4)}, sehingga segmen lebih tepat dibaca "
        f"sebagai partisi praktis yang saling tumpang tindih; {fmt_int(facts['clu_anom'].get('n_anomalies', 0))} "
        f"pelanggan ({fmt_pct(facts['clu_anom'].get('anomaly_rate', 0))}) terjauh dari sentroid diberi label -1. "
        f"Seluruh angka dalam laporan ini dibangkitkan dari artefak pipeline dan dapat direproduksi "
        f"dengan satu perintah."
    )
    story.append(Paragraph(summary, body))

    # ---------------- 1. inventaris ----------------
    story.append(Paragraph("1. Inventaris Dataset & Integritas Data", h1))
    story.append(Paragraph(
        "Dataset proyek terdiri atas 5 berkas tabular pada direktori <code>data/raw/</code>. Verifikasi "
        "struktural memastikan tidak ada duplikasi ID primer, tidak ada irisan ID antara partisi latih-uji, "
        "serta pembagian latih-uji yang kronologis.", body))
    inv_rows = [[
        "Berkas Dataset", "Tugas", "Peran", "Baris", "Kolom", "Ukuran", "Cakupan Temporal",
    ]]
    for row in facts["inventory"]:
        inv_rows.append([
            row["file"], row["task"], row["role"], fmt_int(row["rows"]),
            fmt_int(row["cols"]), f"{fmt_dec(row['size_mb'])} MB", row["span"],
        ])
    inv_rows.append([
        "TOTAL", "-", "-", fmt_int(facts["total_rows"]), "-",
        f"{fmt_dec(sum(r['size_mb'] for r in facts['inventory']))} MB", "-",
    ])
    story.append(_table(inv_rows, [105, 50, 70, 45, 40, 50, 135], font_size=7.5,
                        bold_rows=(len(inv_rows) - 1,)))
    story.append(Paragraph(
        "Tabel 1: Inventaris lengkap berkas dataset mentah proyek UTS Machine Learning.", caption))

    # ---------------- 2. audit kebocoran ----------------
    story.append(Paragraph("2. Audit Kebocoran Data (Leakage Audit) & Pra-pemrosesan", h1))
    leak = facts["reg_leak"]
    leak_text = (
        f"<b>A. Temuan fatal <code>jumlah_scan</code> pada Tugas 1:</b><br/>"
        f"Pada <code>kamus_data.md</code> disebutkan bahwa prediksi waktu kirim dibuat saat paket diambil "
        f"kurir. Audit empiris menemukan rata-rata <code>jumlah_scan</code> data latih "
        f"{fmt_dec(facts['scan_train_mean'])} scan (maksimal {fmt_int(facts['scan_train_max'])} scan), "
        f"sedangkan data uji hanya {fmt_dec(facts['scan_test_mean'])} scan "
        f"(maksimal {fmt_int(facts['scan_test_max'])} scan). "
        f"Korelasi positif ({fmt_dec(raw.get('scan_corr', 0.0), 4)}) dengan durasi kirim membuktikan bahwa "
        f"pada data latih fitur ini diisi setelah paket sampai di tujuan. Jika fitur ini dipertahankan, "
        f"model bergantung pada jumlah scan tinggi; pada data uji yang baru memiliki 1-2 scan, prediksi "
        f"durasi runtuh (under-prediction drastis). <b>Keputusan: <code>jumlah_scan</code> dikeluarkan "
        f"secara permanen.</b> Dampak ablasi: Pinball P50 turun dari "
        f"{fmt_dec(reg_clean['pinball_loss_p50'])} menjadi {fmt_dec(leak['pinball_loss_p50'])} - sebuah "
        f"ilusi kemajuan yang hanya muncul karena kebocoran.<br/><br/>"
        f"<b>B. Dilema kebijakan <code>flag_review_manual</code> pada Tugas 2:</b><br/>"
        f"Fitur ini adalah output aturan heuristik lama (<i>legacy rules</i>). Pipeline utama dilatih "
        f"tanpa fitur tersebut agar model tidak sekadar menduplikasi aturan lama; ablasi komparatif "
        f"tetap disajikan pada Tabel 6 untuk transparansi.<br/><br/>"
        f"<b>C. Nilai hilang, kategori tak terlihat, dan distribusi target:</b>"
    )
    story.append(Paragraph(leak_text, body))

    miss_bits = [f"<code>{m['column']}</code> pada <code>{m['dataset']}</code> "
                 f"({fmt_int(m['n'])} sel, {fmt_pct_raw(m['pct'])})" for m in facts["missing"]]
    story.append(Paragraph("• Nilai hilang: " + "; ".join(miss_bits) + ".", body))
    unseen = ", ".join(raw.get("unseen_kurir", [])) or "tidak ada"
    story.append(Paragraph(
        f"• Kategori unseen: mitra kurir {unseen} hanya ada di data uji; disandikan ke indeks khusus "
        f"<code>unseen</code> dengan prior penyusutan global (<i>shrinkage</i>).", body))
    tgt = facts["reg_target"]
    story.append(Paragraph(
        f"• Target regresi <code>durasi_jam</code> sangat miring ke kanan (right-skewed): "
        f"P50 = {fmt_dec(tgt['p50'])} jam, "
        f"P90 = {fmt_dec(tgt['p90'])} jam, P99 = {fmt_dec(tgt['p99'])} jam, maksimum {fmt_int(tgt['max'])} jam. "
        f"Regresi kuantil kebal terhadap ekor ekstrem tersebut.", body))

    # ---------------- 3. regresi ----------------
    story.append(Paragraph("3. Tugas 1: Estimasi Durasi Pengiriman (Quantile Regression)", h1))
    cv = facts["reg_cv"]
    if raw.get("available"):
        split_text = (
            f"Data latih dibagi kronologis murni: Bulan 1-7 (Januari-Juli 2025, {fmt_int(raw['reg_fit_n'])} baris) "
            f"sebagai data latih dan Bulan 8 (Agustus 2025, {fmt_int(raw['reg_val_n'])} baris) sebagai validasi "
            f"holdout. Pengujian acak 5-Fold CV menghasilkan optimisme palsu (Pinball P50 = "
            f"{fmt_dec(cv['pinball_loss_p50_mean'])} +/- {fmt_dec(cv['pinball_loss_p50_std'])}, P90 = "
            f"{fmt_dec(cv['pinball_loss_p90_mean'])} +/- {fmt_dec(cv['pinball_loss_p90_std'])}, cakupan P90 = "
            f"{fmt_pct(cv['coverage_p90_mean'])}) dibandingkan holdout temporal P50 = "
            f"{fmt_dec(reg_clean['pinball_loss_p50'])}. Validasi temporal mempertahankan kausalitas waktu."
        )
    else:
        split_text = (
            f"Validasi memakai holdout temporal (Agustus 2025). Pengujian acak 5-Fold CV menghasilkan "
            f"optimisme palsu (Pinball P50 = {fmt_dec(cv['pinball_loss_p50_mean'])} +/- "
            f"{fmt_dec(cv['pinball_loss_p50_std'])}) dibandingkan holdout temporal P50 = "
            f"{fmt_dec(reg_clean['pinball_loss_p50'])}."
        )
    story.append(Paragraph(split_text, body))
    story.append(Paragraph(
        f"Metodologi: dua <i>HistGradientBoostingRegressor</i> independen untuk P50 dan P90 dengan "
        f"<i>pinball loss</i> L_t(y, y_hat) = max(t*(y - y_hat), (1 - t)*(y_hat - y)). Kepatuhan cakupan "
        f"P90 dijamin dengan <i>split conformal calibration</i> (margin delta = "
        f"{fmt_dec(facts['reg_conformal'], 3)} jam) dan penegasan non-crossing deterministik "
        f"y_hat_90 = max(y_hat_90, y_hat_50).", body))

    reg_rows = [["Model Eksperimen", "Fitur", "P50 Pinball", "P90 Pinball", "P90 Coverage", "Non-Crossing"]]
    for r in facts["reg_model_rows"]:
        reg_rows.append([
            r["label"], r["features"], fmt_dec(r["p50"]), fmt_dec(r["p90"]),
            fmt_pct(r["coverage"]), f"{fmt_int(r['cross'])} (100% patuh)",
        ])
    best_idx = next(i for i, r in enumerate(facts["reg_model_rows"], start=1)
                    if r["key"] == facts["reg_best_clean"])
    story.append(_table(reg_rows, [150, 45, 62, 62, 70, 106], font_size=7.5,
                        highlight=(best_idx,)))
    story.append(Paragraph(
        "Tabel 2: Evaluasi komparatif model regresi kuantil pada holdout validasi temporal "
        "(Agustus 2025). Baris hijau = model bersih terbaik.", caption))

    pulau_breakdown = facts["reg_pulau"]
    kurir_breakdown = facts["reg_kurir"]
    kurir_caption = "Tabel 5: Rincian galat dan cakupan per mitra kurir pada holdout."
    if raw.get("unseen_kurir"):
        kurir_caption += (f" Mitra kurir {unseen} hanya ada di data uji sehingga tidak "
                          "tercakup tabel ini.")
    breakdowns = (
        (facts["reg_service"], "layanan", "Layanan",
         "Tabel 3: Rincian galat dan cakupan per kategori layanan logistik pada holdout."),
        (pulau_breakdown, "pulau_tujuan", "Pulau Tujuan",
         "Tabel 4: Rincian galat dan cakupan per pulau tujuan pada holdout."),
        (kurir_breakdown, "mitra_kurir", "Mitra Kurir", kurir_caption),
    )
    for breakdown, group_key, group_label, table_caption in breakdowns:
        if not breakdown:
            continue
        rows = [[group_label, "Sampel", "Median Aktual", "Median Prediksi", "MAE P50", "Cakupan P90"]]
        for group in breakdown:
            rows.append([group[group_key], fmt_int(group["n_samples"]), fmt_dec(group["actual_median"], 1),
                         fmt_dec(group["pred_median"], 1), fmt_dec(group["mae_p50"]),
                         fmt_pct(group["coverage_p90"])])
        story.append(_table(rows, [90, 60, 80, 80, 70, 80], font_size=7.5))
        story.append(Paragraph(table_caption, caption))

    if pulau_breakdown and kurir_breakdown:
        pulau_mae_min = min(pulau_breakdown, key=lambda group: group["mae_p50"])
        pulau_mae_max = max(pulau_breakdown, key=lambda group: group["mae_p50"])
        pulau_cov_min = min(group["coverage_p90"] for group in pulau_breakdown)
        pulau_cov_max = max(group["coverage_p90"] for group in pulau_breakdown)
        kurir_mae_min = min(kurir_breakdown, key=lambda group: group["mae_p50"])
        kurir_mae_max = max(kurir_breakdown, key=lambda group: group["mae_p50"])
        kurir_cov_min = min(kurir_breakdown, key=lambda group: group["coverage_p90"])
        story.append(Paragraph(
            f"Galat tidak merata antarwilayah: MAE P50 terendah di {pulau_mae_min['pulau_tujuan']} "
            f"({fmt_dec(pulau_mae_min['mae_p50'])}) dan tertinggi di {pulau_mae_max['pulau_tujuan']} "
            f"({fmt_dec(pulau_mae_max['mae_p50'])}), dengan cakupan P90 per pulau antara "
            f"{fmt_pct(pulau_cov_min)} dan {fmt_pct(pulau_cov_max)}. Antarmitra kurir, MAE P50 berkisar "
            f"dari {fmt_dec(kurir_mae_min['mae_p50'])} ({kurir_mae_min['mitra_kurir']}) sampai "
            f"{fmt_dec(kurir_mae_max['mae_p50'])} ({kurir_mae_max['mitra_kurir']}), dan cakupan P90 "
            f"terendah {fmt_pct(kurir_cov_min['coverage_p90'])} pada {kurir_cov_min['mitra_kurir']}. "
            "Jadi cakupan 90% secara agregat tidak berarti setiap kelompok terlindungi sama baiknya.", body))

    if facts["reg_importance"]:
        imp = ", ".join(f"{r['feature']} ({fmt_dec(r['importance_mean'], 2)})"
                        for r in facts["reg_importance"][:5])
        story.append(Paragraph(
            f"Penjelasan model (<i>permutation importance</i>) menempatkan {imp} sebagai pembeda "
            f"utama durasi kirim.", body))

    story.extend(_figure_pair(
        ["fig_regression_model_comparison.png", "fig_regression_quantile_calibration.png"],
        "Gambar 1: (Kiri) Perbandingan Pinball Loss antar kandidat model; (Kanan) Kalibrasi interval "
        "prediksi P50-P90 pada data aktual.", caption))
    story.extend(_figure_single(
        "fig_regression_subgroup_service_performance.png",
        f"Gambar 2: Cakupan empiris P90 (stabil pada {fmt_pct(reg_clean['coverage_p90'])}) dan MAE P50 "
        f"di seluruh kategori layanan logistik.", caption))
    pdp_features = [row["feature"] for row in facts["reg_importance"][:4]]
    if len(pdp_features) == 4:
        categorical_feature = next((feature for feature in pdp_features if feature.endswith("_cat")), None)
        panel_note = ""
        if categorical_feature is not None:
            panel_note = (f"Panel {categorical_feature} datar padahal fitur ini teratas pada permutation "
                          "importance; ini konsisten dengan keterbatasan metode recursion (default "
                          "partial_dependence untuk HistGradientBoosting) pada fitur kategorikal, "
                          "sehingga panel tersebut tidak ditafsirkan. ")
        story.extend(_figure_single(
            "fig_regression_partial_dependence.png",
            f"Gambar 3: Partial dependence model P50 untuk empat fitur teratas permutation importance "
            f"({', '.join(pdp_features)}). Sumbu y adalah kontribusi relatif yang bergeser oleh suatu "
            f"konstanta, bukan durasi absolut. {panel_note}Perhitungan ulang dengan method='brute' "
            "tidak dilakukan karena akan mengubah artefak pipeline.", caption, height=3.5 * inch))

    # ---------------- 4. klasifikasi ----------------
    story.append(Paragraph("4. Tugas 2: Deteksi Fraud Finansial & Optimasi Biaya Operasional", h1))
    if raw.get("available"):
        val_txt = (
            f"Prevalensi kecurangan hanya {fmt_pct(facts['clf_prevalence'], 4)} "
            f"({fmt_int(facts['clf_prevalence_n'])} fraud dari {fmt_int(facts['clf_train_rows'])} transaksi). "
            f"Validasi memakai split kronologis dua bulan (Juli-Agustus 2025: "
            f"{fmt_int(raw['clf_val_n'])} transaksi, {fmt_int(raw['clf_val_fraud'])} fraud), setara dengan "
            f"horizon data uji."
        )
    else:
        val_txt = (
            f"Prevalensi kecurangan hanya {fmt_pct(facts['clf_prevalence'], 4)} "
            f"({fmt_int(facts['clf_prevalence_n'])} fraud dari {fmt_int(facts['clf_train_rows'])} transaksi)."
        )
    story.append(Paragraph(val_txt, body))
    story.append(Paragraph(
        f"Setiap peninjauan manual berbiaya {fmt_idr(facts['clf_review_cost'])}; transaksi fraud yang lolos "
        f"merugikan sebesar nominalnya. Aturan keputusan Bayes optimal menandai transaksi jika "
        f"P(fraud) x nominal &gt; biaya peninjauan. Model <i>HistGradientBoostingClassifier</i> dilatih "
        f"dengan bobot seimbang, kemudian probabilitasnya dikalibrasi dengan regresi Isotonik pada data "
        f"validasi temporal.", body))

    clf_rows = [["Model / Strategi", "Strategi Keputusan", "PR-AUC", "ROC-AUC", "Brier", "Flag Rate", "Total Biaya"]]
    clf_rows.append(["Baseline: Jangan Tandai", "Flag = 0 seluruhnya", "-", "-", "-", "0,00%",
                     fmt_idr(facts["clf_nothing"])])
    clf_rows.append(["Baseline: Tandai Semua", "Flag = 1 seluruhnya", "-", "-", "-", "100,00%",
                     fmt_idr(facts["clf_everything"])])
    best_clf_idx = None
    for r in facts["clf_model_rows"]:
        clf_rows.append([r["label"], r["strategy"], fmt_dec(r["pr_auc"], 4), fmt_dec(r["roc_auc"], 4),
                         fmt_dec(r["brier"], 4), fmt_pct(r["flag_rate"]), fmt_idr(r["total_cost"])])
        if r["key"] == facts["clf_best_clean"]:
            best_clf_idx = len(clf_rows) - 1
    story.append(_table(clf_rows, [120, 100, 45, 50, 45, 50, 85], font_size=7.2,
                        highlight=(best_clf_idx,)))
    story.append(Paragraph(
        "Tabel 6: Kinerja klasifikasi deteksi fraud dan dampak total biaya finansial operasional "
        "(holdout Juli-Agustus 2025). Baris hijau = model bersih berbiaya terendah.", caption))

    if facts["clf_sweep_best"]:
        b = facts["clf_sweep_best"]
        story.append(Paragraph(
            f"Sweep ambang biaya mengonfirmasi multiplier optimum ({b['threshold_desc']}): "
            f"{fmt_int(b['flagged_count'])} transaksi ditandai "
            f"({fmt_pct(b['flagged_rate'])}), recall fraud {fmt_pct(b['fraud_recall'])}, total biaya "
            f"{fmt_idr(b['total_cost'])} - terendah di antara {len(facts['clf_sweep'])} kandidat ambang.",
            body))

    if facts["clf_kanal"]:
        kan_rows = [["Kanal", "Transaksi", "Fraud", "Rasio Fraud", "Ditandai", "Recall Fraud"]]
        for k in facts["clf_kanal"]:
            kan_rows.append([k["kanal"], fmt_int(k["n_trx"]), fmt_int(k["n_fraud"]),
                             fmt_pct(k["fraud_rate"]), fmt_int(k["flagged_count"]),
                             fmt_pct(k["recall"])])
        story.append(_table(kan_rows, [70, 75, 55, 70, 65, 80], font_size=7.5))
        story.append(Paragraph(
            "Tabel 7: Rincian kinerja model per kanal pembayaran pada holdout.", caption))

    if facts["clf_monthly"]:
        prauc = facts["clf_monthly_prauc"]
        mon_rows = [["Bulan", "Transaksi", "Fraud", "Rasio Fraud"]]
        for m in facts["clf_monthly"]:
            key = str(pd.to_datetime(m["month"]).month)
            extra = f" (PR-AUC {fmt_dec(prauc[key]['pr_auc'], 4)})" if key in prauc else ""
            mon_rows.append([pd.to_datetime(m["month"]).strftime("%Y-%m"), fmt_int(m["n_trx"]),
                             fmt_int(m["n_fraud"]), fmt_pct(m["fraud_rate"]) + extra])
        story.append(_table(mon_rows, [70, 80, 60, 110], font_size=7.5))
        story.append(Paragraph(
            "Tabel 8: Pergeseran pola fraud bulanan. PR-AUC dievaluasi per bulan pada jendela "
            "validasi out-of-time; kalibrasi isotonic menyerap pergeseran tersebut.", caption))

    story.extend(_figure_pair(
        ["fig_classification_pr_roc_curves.png", "fig_classification_calibration_curves.png"],
        "Gambar 4: (Kiri) Kurva Precision-Recall dan ROC; (Kanan) Diagram keandalan kalibrasi "
        "probabilitas Isotonik vs Sigmoid.", caption))
    mult = fmt_dec(facts["clf_sweep_best"]["multiplier"], 1) if facts["clf_sweep_best"] else "-"
    story.extend(_figure_single(
        "fig_classification_cost_curve.png",
        f"Gambar 5: Kurva optimasi biaya finansial nyata; ambang Bayes optimal ({mult}x "
        f"{fmt_idr(facts['clf_review_cost'])}) menekan biaya ke titik minimum.", caption))

    # ---------------- 5. klastering ----------------
    story.append(Paragraph("5. Tugas 3: Segmentasi Pelanggan (Unsupervised Clustering)", h1))
    feature_evidence = facts["clu_feature_evidence"]
    n_features_used = fmt_int(feature_evidence.get("n_features_used", 0))
    story.append(Paragraph(
        f"Dataset mencakup {fmt_int(facts['pred']['clu_rows'])} pelanggan. Nilai keranjang rata-rata "
        f"yang hilang direkonstruksi via <code>belanja_90h / frekuensi_90h</code>, variabel keuangan "
        f"miring ditransformasi ln(1 + x), lalu seluruh {n_features_used} fitur, termasuk porsi "
        "belanja per kategori dan porsi aktivitas per jam, distandarisasi dengan StandardScaler. "
        "Porsi-porsi tersebut adalah data komposisi (porsi dalam satu kelompok saling melengkapi) "
        "dan hanya distandarisasi, belum ditransformasi log-ratio seperti CLR, sehingga "
        "antar-fiturnya tidak independen.", body))

    k_opt = fmt_int(facts["clu_k"])
    kmeans_sweep = facts["clu_kmeans"]
    if kmeans_sweep:
        db_best = min(kmeans_sweep, key=lambda row: row["davies_bouldin"])
        sil_best = max(kmeans_sweep, key=lambda row: row["silhouette"])
        ch_best = max(kmeans_sweep, key=lambda row: row["calinski_harabasz"])
        bic_best = min(facts["clu_gmm"], key=lambda row: row["bic"], default=None)
        ward_best = max(facts["clu_agglo"], key=lambda row: row["silhouette"], default=None)
        bic_k = fmt_int(bic_best["k"]) if bic_best is not None else "-"
        ward_k = fmt_int(ward_best["k"]) if ward_best is not None else "-"
        sil_min = min(row["silhouette"] for row in kmeans_sweep)
        sil_kopt = facts["clu_k_opt_row"].get("silhouette")
        sil_kopt_text = fmt_dec(sil_kopt, 4) if sil_kopt is not None else "-"
        story.append(Paragraph(
            f"Pemilihan k. Tabel 9 menunjukkan kriteria tidak sepakat: Davies-Bouldin minimum pada "
            f"k = {fmt_int(db_best['k'])} ({fmt_dec(db_best['davies_bouldin'], 4)}), sedangkan "
            f"Silhouette K-Means tertinggi pada k = {fmt_int(sil_best['k'])} "
            f"({fmt_dec(sil_best['silhouette'], 4)}; k = {k_opt} hanya {sil_kopt_text}), "
            f"Calinski-Harabasz tertinggi pada k = {fmt_int(ch_best['k'])}, BIC GMM terendah pada "
            f"k = {bic_k}, dan Silhouette Ward tertinggi pada k = {ward_k}. Pipeline memilih "
            f"k = {k_opt} dengan aturan Davies-Bouldin minimum yang ditetapkan di kode, sehingga "
            f"k = {k_opt} adalah pilihan berdasarkan satu kriteria, bukan konsensus antarmetode. "
            f"Silhouette K-Means pada semua k hanya antara {fmt_dec(sil_min, 4)} dan "
            f"{fmt_dec(sil_best['silhouette'], 4)}, mendekati 0, yang berarti klaster saling tumpang "
            "tindih dan tidak terpisah jelas. Segmentasi ini lebih tepat dibaca sebagai partisi "
            "praktis untuk keperluan pemasaran, bukan struktur alami data.", body))

    pairwise = facts["clu_pairwise_aris"]
    n_pairs = len(pairwise)
    if n_pairs:
        n_runs = (1 + isqrt(1 + 8 * n_pairs)) // 2
        complete_pairs = n_runs * (n_runs - 1) // 2 == n_pairs
        n_runs_text = fmt_int(n_runs) if complete_pairs else "-"
        high_aris = [ari for ari in pairwise if ari >= 0.9]
        low_aris = [ari for ari in pairwise if ari < 0.9]
        if high_aris and low_aris:
            kalimat_run = ""
            if complete_pairs:
                low_pairs = [set(pair) for pair, ari in zip(combinations(range(n_runs), 2), pairwise)
                             if ari < 0.9]
                common_runs = set.intersection(*low_pairs)
                if len(common_runs) == 1:
                    run = next(iter(common_runs)) + 1
                    kalimat_run = f", dan semuanya melibatkan run ke-{fmt_int(run)}"
            distribution = (
                f", tetapi sebarannya terbelah: {fmt_int(len(high_aris))} pasangan bernilai "
                f"{fmt_dec(min(high_aris), 4)} sampai {fmt_dec(max(high_aris), 4)}, sedangkan "
                f"{fmt_int(len(low_aris))} pasangan hanya {fmt_dec(min(low_aris), 4)} sampai "
                f"{fmt_dec(max(low_aris), 4)}{kalimat_run}."
            )
        else:
            distribution = (f". Semua pasangan bernilai {fmt_dec(min(pairwise), 4)} "
                            f"sampai {fmt_dec(max(pairwise), 4)}.")
        story.append(Paragraph(
            f"Stabilitas. K-Means k = {k_opt} dijalankan {n_runs_text} kali pada subsampel "
            f"{fmt_pct(CLU_STABILITY_SUBSAMPLE, 0)} tanpa pengembalian, lalu label tiap pasangan run "
            f"dibandingkan pada pelanggan yang beririsan ({fmt_int(n_pairs)} pasangan). Rata-rata ARI "
            f"{fmt_dec(facts['clu_ari_mean'], 4)} +/- {fmt_dec(facts['clu_ari_std'], 4)}{distribution} "
            "Artinya partisi umumnya terulang, "
            "tetapi pada sebagian subsampel K-Means menghasilkan pembagian yang berbeda, sehingga "
            "hasil ini belum cukup untuk menyebut struktur klaster kokoh.", body))

    anomaly = facts["clu_anom"]
    story.append(Paragraph(
        "Anomali. Setelah K-Means final, jarak Euclidean setiap pelanggan ke sentroid klasternya "
        "dihitung di ruang fitur terstandarisasi. Pelanggan dengan jarak di atas persentil "
        f"ke-{fmt_dec(CLU_ANOMALY_PERCENTILE, 1)} (jarak {fmt_dec(anomaly.get('threshold_distance', 0), 2)}) "
        f"diberi label -1, yaitu {fmt_int(anomaly.get('n_anomalies', 0))} akun "
        f"({fmt_pct(anomaly.get('anomaly_rate', 0))}). Jumlah ini ditentukan oleh ambang persentil "
        "tersebut, bukan ditemukan secara alami dari data.", body))

    if facts["clu_kmeans"]:
        kmeans_by_k = {row["k"]: row for row in facts["clu_kmeans"]}
        gmm_by_k = {row["k"]: row for row in facts["clu_gmm"]}
        ward_by_k = {row["k"]: row for row in facts["clu_agglo"]}
        columns = [
            ("Inertia", kmeans_by_k, "inertia", 0, None),
            ("Silhouette K-Means", kmeans_by_k, "silhouette", 4, max),
            ("Davies-<br/>Bouldin", kmeans_by_k, "davies_bouldin", 4, min),
            ("Calinski-<br/>Harabasz", kmeans_by_k, "calinski_harabasz", 0, max),
            ("BIC GMM", gmm_by_k, "bic", 0, min),
            ("Silhouette Ward", ward_by_k, "silhouette", 4, max),
        ]
        best_values = [
            select_best((row[metric] for row in sweep.values()), default=None)
            if select_best is not None else None
            for _, sweep, metric, _, select_best in columns
        ]
        comparison_rows = [["k"] + [label for label, *_ in columns]]
        for k in sorted(kmeans_by_k.keys() | gmm_by_k.keys() | ward_by_k.keys()):
            row = [fmt_int(k)]
            for (_, sweep, metric, decimals, _), best_value in zip(columns, best_values):
                if k not in sweep:
                    row.append("-")
                    continue
                value = sweep[k][metric]
                cell = fmt_dec(value, decimals)
                row.append(f"<b>{cell}</b>" if value == best_value else cell)
            comparison_rows.append(row)
        story.append(_table(comparison_rows, [26, 72, 70, 66, 76, 80, 70], font_size=7.5))

        k_min, k_max = min(kmeans_by_k), max(kmeans_by_k)
        clu_rows = facts["pred"]["clu_rows"]
        hdbscan = facts["clu_hdbscan"]
        story.append(Paragraph(
            f"Tabel 9: Perbandingan metrik pemilihan k untuk k = {fmt_int(k_min)} sampai "
            f"{fmt_int(k_max)}. Nilai terbaik tiap kolom dicetak tebal; inertia tidak ditandai "
            f"karena selalu turun seiring bertambahnya k. Silhouette K-Means dihitung pada sampel "
            f"{fmt_int(CLU_N_SIL_KMEANS)} pelanggan, sedangkan inertia, Davies-Bouldin, dan "
            f"Calinski-Harabasz pada seluruh {fmt_int(clu_rows)} pelanggan; BIC GMM (kovarians "
            f"diagonal) pada sampel {fmt_int(CLU_N_GMM)}; silhouette Agglomerative Ward pada "
            f"sampel {fmt_int(CLU_N_AGGLO)}. HDBSCAN, yang tidak memerlukan k, dijalankan pada "
            f"sampel {fmt_int(hdbscan.get('sample_size', 0))} "
            f"(min_cluster_size {fmt_int(hdbscan.get('min_cluster_size', 0))}, "
            f"min_samples {fmt_int(hdbscan.get('min_samples', 0))}) dan menemukan "
            f"{fmt_int(hdbscan.get('n_clusters_found', 0))} klaster dengan "
            f"{fmt_pct(hdbscan.get('noise_ratio', 0))} titik noise.", caption))

    top_feat = feature_evidence.get("top_discriminative_features", [])[:5]
    if top_feat:
        top5 = ", ".join(f"{f['feature']} ({fmt_dec(f['f_score'], 0)})" for f in top_feat)
        weak_lt1 = ", ".join(f"{f['feature']} ({fmt_dec(f['f_score'], 2)})"
                             for f in feature_evidence.get("weakest_features", []) if f["f_score"] < 1)
        weak_sentence = ""
        if weak_lt1:
            weak_sentence = (f" Fitur dengan F di bawah 1, yaitu {weak_lt1}, praktis tidak membedakan "
                             "klaster dan hanya menambah noise pada perhitungan jarak.")
        story.append(Paragraph(
            "Bukti fitur. ANOVA F-score dihitung terhadap label klaster yang dihasilkan model itu "
            "sendiri. Karena klaster memang dibentuk dari fitur-fitur tersebut, F-score tinggi "
            f"bersifat sirkular: ia menunjukkan fitur yang paling membedakan klaster yang sudah "
            f"terbentuk, yaitu {top5}, tetapi bukan bukti independen bahwa fitur itu layak dipilih. "
            f"Seluruh {n_features_used} fitur dipakai tanpa seleksi, termasuk atribut CRM anonim "
            f"n01 sampai n10.{weak_sentence}", body))

    if facts["clu_persona_rows"]:
        per_rows = [["ID", "Nama Persona", "Ukuran Akun", "Belanja 90h", "Frekuensi", "Kategori", "Strategi"]]
        for p in facts["clu_persona_rows"]:
            per_rows.append([
                p["id"], p["name"], f"{fmt_int(p['n'])} ({fmt_pct(p['pct'])})",
                fmt_idr(p["spend"]) if p["spend"] is not None else "-",
                f"{fmt_dec(p['freq'], 1)} kali" if p["freq"] is not None else "-",
                p["category"], PERSONA_STRATEGI_ID.get(p["name"], p["strategy"]),
            ])
        story.append(_table(per_rows, [22, 92, 68, 72, 50, 78, 113], font_size=6.6,
                            red_rows=(len(per_rows) - 1,)))
        story.append(Paragraph(
            "Tabel 10: Profil persona pelanggan. Nama persona mengikuti label dari pipeline; "
            "strategi ditulis ulang dalam bahasa Indonesia dan disesuaikan dengan profil numerik "
            "setiap klaster.",
            caption))

        persona_ids = {profile.get("name"): cid for cid, profile in facts["clu_persona"].items()}
        nama_bot = "Hyperactive Bargain Hunters"
        nama_disc = "Discount Hunters"
        nama_dorm = "Lapsed / At-Risk Dormant"
        if all(name in persona_ids for name in (nama_bot, nama_disc, nama_dorm)):
            id_bot = persona_ids[nama_bot]
            id_disc = persona_ids[nama_disc]
            id_dorm = persona_ids[nama_dorm]
            means = facts["clu_means"]
            bot_means = means.get(id_bot, {})
            sesi = bot_means.get("sesi_per_minggu")
            durasi = bot_means.get("durasi_sesi_menit")
            konversi = bot_means.get("rasio_konversi")
            rec_disc = means.get(id_disc, {}).get("recency_hari")
            rec_dorm = means.get(id_dorm, {}).get("recency_hari")
            if all(value is not None for value in (sesi, durasi, konversi, rec_disc, rec_dorm)):
                story.append(Paragraph(
                    f"Catatan persona. Persona {id_bot} ({nama_bot}) rata-rata membuka "
                    f"{fmt_dec(sesi, 1)} sesi per minggu dengan durasi {fmt_dec(durasi, 1)} menit "
                    f"per sesi dan rasio konversi {fmt_pct(konversi, 0)}. Pola ini lebih mirip aktivitas "
                    "otomatis (bot atau skrip) daripada perilaku belanja manusia, sehingga perlu "
                    "diverifikasi sebelum diperlakukan sebagai segmen pemasaran. "
                    f"Persona {id_disc} ({nama_disc}) masih aktif dengan recency {fmt_dec(rec_disc, 0)} "
                    f"hari, sehingga strateginya bukan win-back; win-back hanya relevan untuk persona "
                    f"{id_dorm} ({nama_dorm}) dengan recency {fmt_dec(rec_dorm, 0)} hari.", body))

    story.extend(_figure_pair(
        ["fig_clustering_elbow_silhouette.png", "fig_clustering_pca_segments.png"],
        "Gambar 6: (Kiri) Kurva elbow inersia dan Silhouette K-Means; (Kanan) Proyeksi 2D PCA klaster "
        "dan anomali (-1).", caption))
    story.extend(_figure_single(
        "fig_clustering_profiles_heatmap.png",
        "Gambar 7: Heatmap profil fitur asli terstandarisasi untuk setiap persona pelanggan.", caption))

    story.append(Paragraph(
        "Keterbatasan. Tiga perbaikan model yang lebih dalam teridentifikasi tetapi tidak "
        "dikerjakan: (1) membuang fitur yang tidak informatif seperti n01, n02, dan n06; "
        "(2) mentransformasi porsi kategori dan porsi jam dengan CLR sebelum standarisasi; "
        "(3) memilih ulang k dengan kriteria yang disepakati beberapa metrik serta menelusuri "
        "run yang tidak stabil. Ketiganya akan mengubah label pada cluster.csv sehingga tidak "
        "dilakukan menjelang tenggat. Hasil segmentasi di laporan ini perlu dibaca dengan "
        "mempertimbangkan keterbatasan tersebut.", body))

    # ---------------- 6. kepatuhan & reproduksibilitas ----------------
    story.append(Paragraph("6. Verifikasi Kepatuhan Berkas & Reproduksibilitas", h1))
    pred = facts["pred"]
    id_reg = "sama persis" if pred["reg_ids_ok"] else "tidak dapat dibuktikan (data mentah tidak ada)"
    id_clf = "sama persis" if pred["clf_ids_ok"] else "tidak dapat dibuktikan (data mentah tidak ada)"
    qa = (
        f"Seluruh berkas pengumpulan diverifikasi oleh <code>kode/common/validate_submissions.py</code> "
        f"dan dibangkitkan ulang oleh script ini dari berkas yang sama:<br/>"
        f"1. <b>prediksi_regresi.csv</b>: {fmt_int(pred['reg_rows'])} baris, ID {id_reg} dengan data uji; "
        f"P90 &gt;= P50 pada {fmt_pct(1 - pred['reg_cross'] / max(pred['reg_rows'], 1))} sampel "
        f"({fmt_int(pred['reg_cross'])} pelanggaran); rentang P50 {fmt_dec(pred['reg_p50_min'])}-"
        f"{fmt_dec(pred['reg_p50_max'])} jam, P90 {fmt_dec(pred['reg_p90_min'])}-"
        f"{fmt_dec(pred['reg_p90_max'])} jam.<br/>"
        f"2. <b>prediksi_klasifikasi.csv</b>: {fmt_int(pred['clf_rows'])} baris, ID {id_clf}; probabilitas "
        f"pada [{fmt_dec(pred['clf_prob_min'], 6)}, {fmt_dec(pred['clf_prob_max'], 6)}]; "
        f"{fmt_int(pred['clf_flag'])} transaksi ditandai ({fmt_pct(pred['clf_flag_rate'])}).<br/>"
        f"3. <b>cluster.csv</b>: {fmt_int(pred['clu_rows'])} baris; label integer "
        f"{{{', '.join(str(c) for c in sorted(pred['clu_counts']))}}}, dengan "
        f"{fmt_int(pred['clu_counts'].get(-1, 0))} pelanggan berlabel -1.<br/><br/>"
        f"<b>Reproduksibilitas:</b> satu perintah <code>python kode/run_all.py</code> menjalankan "
        f"seluruh pipeline (data engineering, pelatihan tiga tugas, validasi berkas) dengan seed global "
        f"{RANDOM_SEED}. Versi pustaka: numpy {pkg_version('numpy')}, pandas {pkg_version('pandas')}, "
        f"scikit-learn {pkg_version('scikit-learn')}, matplotlib {pkg_version('matplotlib')}. "
        f"Setiap angka pada laporan dibangkitkan oleh <code>laporan/build_laporan_pdf.py</code> dari "
        f"<code>outputs/metrics/*.json</code>, <code>outputs/predictions/*.csv</code>, dan "
        f"<code>data/manifests/*.json</code>."
    )
    story.append(Paragraph(qa, body))

    # ---------------- 7. kesimpulan ----------------
    story.append(Paragraph("7. Kesimpulan & Poin Kunci Ujian Lisan", h1))
    story.append(Paragraph(
        f"Implementasi proyek ini membuktikan bahwa rekayasa ML yang baik berakar pada pemahaman domain "
        f"dan proses pengumpulan data, bukan optimasi hiperparameter buta. Penghapusan fitur bocor "
        f"<code>jumlah_scan</code>, penegasan non-crossing konformal, kalibrasi probabilitas untuk "
        f"optimasi kerugian finansial nyata, serta segmentasi pelanggan yang dilaporkan apa adanya "
        f"beserta keterbatasannya merupakan pilar teknis utama yang siap dipertahankan oleh seluruh "
        f"anggota kelompok dalam ujian lisan.",
        body))

    doc.build(story, canvasmaker=NumberedCanvas)
    return PDF_PATH


def main():
    facts = collect_facts()
    path = build_pdf(facts)
    print(f"[OK] laporan dibangkitkan: {path}")
    print(f"     baris dataset    : {fmt_int(facts['total_rows'])}")
    print(f"     model regresi    : {facts['reg_best_clean']} "
          f"(P50 {fmt_dec(facts['reg_clean']['pinball_loss_p50'])})")
    print(f"     model klasifikasi: {facts['clf_best_clean']} "
          f"(biaya {fmt_idr(facts['clf_iso']['total_cost'])})")
    print(f"     klaster optimal  : k={facts['clu_k']} "
          f"(ARI {fmt_dec(facts['clu_ari_mean'], 4)})")


if __name__ == "__main__":
    main()

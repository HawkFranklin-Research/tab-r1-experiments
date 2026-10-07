#!/usr/bin/env python3
"""Formal preprocessing pipeline for local CPTAC/TCGA multiomics cohorts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split


CANCERS = ["BRCA", "LUAD", "LSCC", "HNSCC", "ESCA"]
TCGA_STUDIES = {
    "BRCA": "brca_tcga_pan_can_atlas_2018",
    "LUAD": "luad_tcga_pan_can_atlas_2018",
    "LSCC": "lusc_tcga_pan_can_atlas_2018",
    "HNSCC": "hnsc_tcga_pan_can_atlas_2018",
    "ESCA": "esca_tcga_pan_can_atlas_2018",
}
TCGA_CANCER_CODES = {"LSCC": "LUSC", "HNSCC": "HNSC"}
GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"
NA_VALUES = ["", "NA", "NaN", "nan", "N/A", "null", "NULL", "--"]
CONTINUOUS_MODALITIES = {
    "rnaseq_gene",
    "rnaseq_isoform",
    "circrna",
    "mirna",
    "methylation_gene",
    "cnv_log2",
    "protein_gene",
    "protein_sepep",
    "phosphosite",
    "rppa_gene",
    "rppa_analyte",
    "tcga_protein_gene",
    "tcga_phosphosite",
    "phenotype",
}
BINARY_MODALITIES = {"mutation_binary", "cna_gistic", "cnv_gistic", "mutation_site"}
CORE_MODALITIES = ["rnaseq_gene", "cnv_log2", "cnv_gistic", "mutation_binary", "methylation_gene"]
PROTEOGENOMIC_MODALITIES = CORE_MODALITIES + [
    "protein_gene",
    "protein_sepep",
    "phosphosite",
    "rppa_gene",
    "tcga_protein_gene",
    "tcga_phosphosite",
]
GRAPH_MODALITIES = [
    "rnaseq_gene",
    "methylation_gene",
    "cnv_log2",
    "cnv_gistic",
    "mutation_binary",
    "protein_gene",
    "tcga_protein_gene",
]

OUTLIER_AUDIT_MODALITIES = CONTINUOUS_MODALITIES - {"phenotype"}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_dirs(out_dir: Path) -> None:
    for name in ["qc", "reference", "harmonized", "clinical", "train_ready", "graphs"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        return sum(1 for _ in f)


def normalize_raw_id(value: object) -> str:
    text = str(value).strip()
    text = text.replace(".", "-")
    text = re.sub(r"\s+", "_", text)
    return text


def tcga_patient_from_sample(sample_id: str) -> str:
    sample_id = normalize_raw_id(sample_id)
    return sample_id[:12] if sample_id.startswith("TCGA-") else sample_id


def tcga_tissue(sample_id: str) -> str:
    sample_id = normalize_raw_id(sample_id)
    parts = sample_id.split("-")
    if len(parts) >= 4:
        code = parts[3][:2]
        if code in {"01", "02", "03", "05", "06", "07"}:
            return "tumor"
        if code in {"10", "11", "12", "13", "14"}:
            return "normal"
    return "unknown"


def canonical_patient(cohort: str, cancer: str, raw_patient_id: object) -> str:
    raw = normalize_raw_id(raw_patient_id)
    if cohort.upper().startswith("TCGA"):
        raw = tcga_patient_from_sample(raw)
    return f"{cohort.upper()}:{cancer}:{raw}"


def canonical_sample(cohort: str, cancer: str, raw_sample_id: object) -> str:
    return f"{cohort.upper()}:{cancer}:{normalize_raw_id(raw_sample_id)}"


def parse_gencode_attrs(attr: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in re.findall(r'(\S+) "([^"]+)"', attr):
        out[key] = value
    return out


def strip_ensembl_version(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\.\d+$", "", text)


def normalize_symbol(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return None
    return text.upper()


@dataclass
class GeneMapper:
    ensembl_to_symbol: dict[str, str]

    @classmethod
    def load(cls, out_dir: Path) -> "GeneMapper":
        ref = out_dir / "reference" / "gene_reference.parquet"
        if not ref.exists():
            return cls({})
        df = pd.read_parquet(ref)
        if "ensembl_gene_id" not in df.columns or "gene_symbol" not in df.columns:
            return cls({})
        pairs = df[["ensembl_gene_id", "gene_symbol"]].dropna().drop_duplicates()
        return cls(dict(zip(pairs["ensembl_gene_id"], pairs["gene_symbol"].str.upper())))

    def gene(self, raw: object = None, symbol: object = None, entrez: object = None) -> str:
        sym = normalize_symbol(symbol)
        if sym:
            return f"GENE:{sym}"
        raw_text = "" if raw is None else str(raw).strip()
        if raw_text.startswith("ENSG"):
            ens = strip_ensembl_version(raw_text)
            sym = self.ensembl_to_symbol.get(ens)
            return f"GENE:{sym}" if sym else f"ENSG:{ens}"
        if entrez is not None and str(entrez).strip() not in {"", "NA", "nan"}:
            return f"ENTREZ:{str(entrez).strip()}"
        sym = normalize_symbol(raw_text)
        return f"GENE:{sym}" if sym else f"FEATURE:{raw_text}"


def build_inventory(project_root: Path, out_dir: Path) -> None:
    logging.info("Building raw inventory")
    rows = []
    raw_roots = [project_root / "cptac-5", project_root / "tcga-5"]
    for path in sorted(p for root in raw_roots if root.exists() for p in root.rglob("*") if p.is_file()):
        if path.name.startswith("."):
            continue
        try:
            with path.open("rt", errors="replace") as f:
                first = f.readline().rstrip("\n")
            n_cols = len(first.split("\t")) if first else 0
            rows.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(project_root)),
                    "size_bytes": path.stat().st_size,
                    "suffix": "".join(path.suffixes),
                    "line_count": count_lines(path) if path.stat().st_size < 2_000_000_000 else None,
                    "header_columns": n_cols,
                    "sha256": sha256_file(path),
                }
            )
        except Exception as exc:
            rows.append({"path": str(path), "relative_path": str(path.relative_to(project_root)), "error": str(exc)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "qc" / "raw_inventory.csv", index=False)
    df.to_json(out_dir / "qc" / "raw_inventory.json", orient="records", indent=2)
    logging.info("Inventory rows: %d", len(df))


def build_gene_reference(project_root: Path, out_dir: Path, download: bool = True) -> None:
    logging.info("Building gene reference")
    ref_dir = out_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    gtf_gz = ref_dir / "gencode.v44.annotation.gtf.gz"
    if download and not gtf_gz.exists():
        logging.info("Downloading GENCODE annotation: %s", GENCODE_URL)
        with requests.get(GENCODE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with gtf_gz.open("wb") as f:
                shutil.copyfileobj(r.raw, f)
    rows = []
    if gtf_gz.exists():
        with gzip.open(gtf_gz, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[2] != "gene":
                    continue
                attrs = parse_gencode_attrs(parts[8])
                gene_id = strip_ensembl_version(attrs.get("gene_id", ""))
                gene_name = normalize_symbol(attrs.get("gene_name"))
                if gene_id and gene_name:
                    rows.append(
                        {
                            "ensembl_gene_id": gene_id,
                            "gene_symbol": gene_name,
                            "gene_type": attrs.get("gene_type") or attrs.get("gene_biotype"),
                            "source": "GENCODE_v44",
                        }
                    )
    # Add observed TCGA symbol/Entrez pairs as supplemental evidence.
    for path in sorted((project_root / "tcga-5").glob("*/*/data_mrna_seq_v2_rsem.txt")):
        try:
            tmp = pd.read_csv(path, sep="\t", usecols=["Hugo_Symbol", "Entrez_Gene_Id"], na_values=NA_VALUES)
            tmp = tmp.dropna(subset=["Hugo_Symbol"])
            for _, row in tmp.drop_duplicates().iterrows():
                sym = normalize_symbol(row["Hugo_Symbol"])
                if sym:
                    rows.append(
                        {
                            "ensembl_gene_id": None,
                            "gene_symbol": sym,
                            "entrez_gene_id": row.get("Entrez_Gene_Id"),
                            "source": "TCGA_observed",
                        }
                    )
        except Exception as exc:
            logging.warning("Could not supplement reference from %s: %s", path, exc)
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_parquet(ref_dir / "gene_reference.parquet", index=False)
    df.to_csv(ref_dir / "gene_reference.csv", index=False)
    logging.info("Reference rows: %d", len(df))


def feature_for_modality(modality: str, mapper: GeneMapper, row: pd.Series) -> str:
    if modality in {"rnaseq_gene", "cnv_log2", "cnv_gistic", "mutation_binary", "methylation_gene", "protein_gene"}:
        if "Hugo_Symbol" in row.index:
            return mapper.gene(symbol=row.get("Hugo_Symbol"), entrez=row.get("Entrez_Gene_Id"))
        if "NAME" in row.index:
            return mapper.gene(symbol=row.get("NAME"))
        return mapper.gene(raw=row.iloc[0])
    if modality == "tcga_protein_gene":
        raw = str(row.iloc[0])
        symbol = raw.split("|", 1)[0]
        return mapper.gene(symbol=symbol)
    if modality in {"rppa_gene", "rppa_analyte"}:
        raw = str(row.iloc[0])
        symbol = raw.split("|", 1)[0]
        return mapper.gene(symbol=symbol)
    if modality in {"phosphosite", "tcga_phosphosite"}:
        if "GENE_SYMBOL" in row.index:
            gene = mapper.gene(symbol=row.get("GENE_SYMBOL"))
            site = str(row.get("PHOSPHOSITE", row.iloc[0])).upper()
            return f"PHOS:{gene}:{site}:{str(row.iloc[0])[:80]}"
        raw = str(row.iloc[0])
        parts = raw.split("|")
        gene = mapper.gene(raw=parts[0] if parts else raw)
        site = parts[2] if len(parts) > 2 else raw
        return f"PHOS:{gene}:{site}:{raw[:80]}"
    if modality == "protein_sepep":
        raw = str(row.iloc[0])
        gene = raw.split("_SEPEP", 1)[0]
        return f"SEPEP:{mapper.gene(symbol=gene)}:{raw[:100]}"
    if modality == "mirna":
        return f"MIRNA:{str(row.iloc[0]).strip()}"
    if modality == "circrna":
        return f"CIRC:{str(row.iloc[0]).strip()}"
    if modality == "rnaseq_isoform":
        return f"TX:{str(row.iloc[0]).strip()}"
    if modality == "fusion":
        return f"FUSION:{str(row.iloc[0]).strip()}"
    if modality in {"mutation_site", "cna_focal", "cna_focal_threshold", "phenotype"}:
        return f"{modality.upper()}:{str(row.iloc[0]).strip()}"
    return f"FEATURE:{str(row.iloc[0]).strip()}"


def aggregate_duplicate_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    transposed = df.T
    if method == "max":
        return transposed.groupby(level=0, sort=False).max().T
    return transposed.groupby(level=0, sort=False).median().T


def read_feature_matrix(
    path: Path,
    cohort: str,
    cancer: str,
    modality: str,
    mapper: GeneMapper,
    id_columns: list[str] | None = None,
    tissue_hint: str = "unknown",
    aggregate: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Reading %s %s %s", cohort, cancer, path.name)
    df = pd.read_csv(path, sep="\t", na_values=NA_VALUES, low_memory=False)
    if df.empty:
        raise ValueError(f"Empty matrix: {path}")
    if id_columns is None:
        id_columns = [df.columns[0]]
    id_columns = [col for col in id_columns if col in df.columns]
    sample_cols = [col for col in df.columns if col not in id_columns]
    feature_ids = [feature_for_modality(modality, mapper, row) for _, row in df[id_columns].iterrows()]
    values = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    values.index = feature_ids
    values = aggregate_duplicate_columns(values.T, aggregate)
    values.index = [canonical_sample(cohort, cancer, s) for s in values.index]
    values.index.name = "sample_id"
    feature_meta = pd.DataFrame(
        {
            "feature_id": values.columns,
            "modality": modality,
            "source_path": str(path),
        }
    ).drop_duplicates()
    sample_meta = pd.DataFrame(
        {
            "sample_id": values.index,
            "raw_sample_id": sample_cols,
            "patient_id": [canonical_patient(cohort, cancer, s) for s in sample_cols],
            "cohort": cohort.upper(),
            "cancer": cancer,
            "tissue": [tcga_tissue(s) if cohort.upper().startswith("TCGA") else tissue_hint for s in sample_cols],
            "modality": modality,
        }
    )
    return values.reset_index(), feature_meta, sample_meta


def save_matrix_bundle(out_dir: Path, cohort: str, cancer: str, modality: str, matrix: pd.DataFrame, features: pd.DataFrame, samples: pd.DataFrame) -> None:
    dest = out_dir / "harmonized" / cohort.lower() / cancer
    dest.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(dest / f"{modality}.parquet", index=False)
    features.to_parquet(dest / f"{modality}.features.parquet", index=False)
    samples.to_parquet(dest / f"{modality}.samples.parquet", index=False)


def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make mixed clinical metadata columns safe for Arrow/Parquet serialization."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("string")
    return out


def cptac_specs(cancer: str, root: Path) -> list[tuple[str, Path, list[str] | None, str, str]]:
    d = root / "cptac-5" / "downloads" / cancer
    specs: list[tuple[str, Path, list[str] | None, str, str]] = []
    if cancer == "ESCA":
        linked = [
            ("mirna", "*miRNASeq*RPM_log2.cct", None, "tumor", "median"),
            ("cna_focal", "*GISTIC2.cct", None, "tumor", "median"),
            ("cna_focal_threshold", "*GISTIC2_threshold.cgt", None, "tumor", "max"),
            ("rppa_analyte", "*Analyte*RPPA.cct", None, "tumor", "median"),
            ("rppa_gene", "*Gene*RPPA.cct", None, "tumor", "median"),
            ("mutation_binary", "*Gene*MutSig2CV.cbt", None, "tumor", "max"),
            ("mutation_site", "*Site*MutSig2CV.cbt", None, "tumor", "max"),
        ]
        for modality, pattern, ids, tissue, agg in linked:
            for p in d.glob(pattern):
                specs.append((modality, p, ids, tissue, agg))
        return specs
    patterns = [
        ("rnaseq_gene", f"{cancer}_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt", None, "tumor", "median"),
        ("rnaseq_gene", f"{cancer}_RNAseq_gene_RSEM_coding_UQ_1500_log2_Normal.txt", None, "normal", "median"),
        ("rnaseq_isoform", f"{cancer}_RNAseq_isoform_FPKM_log2_Tumor.txt", None, "tumor", "median"),
        ("rnaseq_isoform", f"{cancer}_RNAseq_isoform_FPKM_log2_Normal.txt", None, "normal", "median"),
        ("circrna", f"{cancer}_RNAseq_circRNA_RSEM_UQ_log2_Tumor.txt", None, "tumor", "median"),
        ("circrna", f"{cancer}_RNAseq_circRNA_RSEM_UQ_log2_Normal.txt", None, "normal", "median"),
        ("mirna", f"{cancer}_miRNAseq_mature_miRNA_RPM_log2_Tumor.txt", None, "tumor", "median"),
        ("mirna", f"{cancer}_miRNAseq_mature_miRNA_RPM_log2_Normal.txt", None, "normal", "median"),
        ("methylation_gene", f"{cancer}_methylation_gene_beta_value_Tumor.txt", None, "tumor", "median"),
        ("methylation_gene", f"{cancer}_methylation_gene_beta_value_Normal.txt", None, "normal", "median"),
        ("cnv_log2", f"{cancer}_WES_CNV_gene_ratio_log2.txt", None, "tumor", "median"),
        ("cnv_gistic", f"{cancer}_WES_CNV_gene_gistic_level.txt", None, "tumor", "max"),
        ("mutation_binary", f"{cancer}_somatic_mutation_gene_level_binary.txt", None, "tumor", "max"),
        ("protein_gene", f"{cancer}_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt", None, "tumor", "median"),
        ("protein_gene", f"{cancer}_proteomics_gene_abundance_log2_reference_intensity_normalized_Normal.txt", None, "normal", "median"),
        ("phosphosite", f"{cancer}_phospho_site_abundance_log2_reference_intensity_normalized_Tumor.txt", None, "tumor", "median"),
        ("phosphosite", f"{cancer}_phospho_site_abundance_log2_reference_intensity_normalized_Normal.txt", None, "normal", "median"),
        ("protein_sepep", f"CPTAC_pancan_{cancer}_proteomics_SEPEP_log2_ratio_normalized_Tumor.cct", None, "tumor", "median"),
        ("protein_sepep", f"CPTAC_pancan_{cancer}_proteomics_SEPEP_log2_ratio_normalized_Normal.cct", None, "normal", "median"),
        ("fusion", f"{cancer}_RNAseq_gene_fusion_Tumor.txt", None, "tumor", "max"),
    ]
    for modality, filename, ids, tissue, agg in patterns:
        p = d / filename
        if p.exists():
            specs.append((modality, p, ids, tissue, agg))
    return specs


def tcga_specs(cancer: str, root: Path) -> list[tuple[str, Path, list[str] | None, str, str]]:
    study = TCGA_STUDIES[cancer]
    d = root / "tcga-5" / study / study
    patterns = [
        ("rnaseq_gene", "data_mrna_seq_v2_rsem.txt", ["Hugo_Symbol", "Entrez_Gene_Id"], "unknown", "median"),
        ("cnv_gistic", "data_cna.txt", ["Hugo_Symbol", "Entrez_Gene_Id"], "unknown", "max"),
        ("cnv_log2", "data_log2_cna.txt", ["Hugo_Symbol", "Entrez_Gene_Id"], "unknown", "median"),
        ("methylation_gene", "data_methylation_hm27_hm450_merged.txt", ["ENTITY_STABLE_ID", "NAME", "DESCRIPTION", "TRANSCRIPT_ID"], "unknown", "median"),
        ("rppa_gene", "data_rppa.txt", ["Composite.Element.REF"], "unknown", "median"),
        ("rppa_gene", "data_rppa_zscores.txt", ["Composite.Element.REF"], "unknown", "median"),
        ("tcga_protein_gene", "data_protein_quantification.txt", ["Composite.Element.REF"], "unknown", "median"),
        ("tcga_phosphosite", "data_phosphoprotein_quantification.txt", ["ENTITY_STABLE_ID", "NAME", "DESCRIPTION", "GENE_SYMBOL", "PHOSPHOSITE", "ATTRIBUTE_NAME"], "unknown", "median"),
    ]
    return [(m, d / f, ids, tissue, agg) for m, f, ids, tissue, agg in patterns if (d / f).exists()]


def derive_tcga_mutation_binary(path: Path, cohort: str, cancer: str, mapper: GeneMapper) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Deriving mutation binary from %s", path)
    usecols = ["Hugo_Symbol", "Entrez_Gene_Id", "Tumor_Sample_Barcode"]
    maf = pd.read_csv(path, sep="\t", usecols=usecols, na_values=NA_VALUES, low_memory=False)
    maf = maf.dropna(subset=["Tumor_Sample_Barcode"])
    maf["feature_id"] = [mapper.gene(symbol=s, entrez=e) for s, e in zip(maf["Hugo_Symbol"], maf["Entrez_Gene_Id"])]
    maf["sample_id"] = [canonical_sample(cohort, cancer, s) for s in maf["Tumor_Sample_Barcode"]]
    maf = maf[["sample_id", "feature_id"]].drop_duplicates()
    maf["value"] = 1
    matrix = maf.pivot_table(index="sample_id", columns="feature_id", values="value", aggfunc="max", fill_value=0)
    matrix.index.name = "sample_id"
    features = pd.DataFrame({"feature_id": matrix.columns, "modality": "mutation_binary", "source_path": str(path)})
    samples = pd.DataFrame(
        {
            "sample_id": matrix.index,
            "raw_sample_id": [x.split(":", 2)[2] for x in matrix.index],
            "patient_id": [canonical_patient(cohort, cancer, x.split(":", 2)[2]) for x in matrix.index],
            "cohort": cohort.upper(),
            "cancer": cancer,
            "tissue": [tcga_tissue(x.split(":", 2)[2]) for x in matrix.index],
            "modality": "mutation_binary",
        }
    )
    return matrix.reset_index(), features, samples


def harmonize_omics(project_root: Path, out_dir: Path) -> None:
    mapper = GeneMapper.load(out_dir)
    sample_frames = []
    feature_frames = []
    failures = []
    for cancer in CANCERS:
        for modality, path, ids, tissue, agg in cptac_specs(cancer, project_root):
            try:
                matrix, features, samples = read_feature_matrix(path, "CPTAC", cancer, modality, mapper, ids, tissue, agg)
                save_matrix_bundle(out_dir, "cptac", cancer, modality, matrix, features, samples)
                sample_frames.append(samples)
                feature_frames.append(features.assign(cohort="CPTAC", cancer=cancer))
            except Exception as exc:
                logging.exception("Failed CPTAC %s %s %s", cancer, modality, path)
                failures.append({"cohort": "CPTAC", "cancer": cancer, "modality": modality, "path": str(path), "error": str(exc)})
        for modality, path, ids, tissue, agg in tcga_specs(cancer, project_root):
            try:
                matrix, features, samples = read_feature_matrix(path, "TCGA", cancer, modality, mapper, ids, tissue, agg)
                if modality == "rnaseq_gene":
                    sample_cols = [c for c in matrix.columns if c != "sample_id"]
                    matrix[sample_cols] = np.log2(matrix[sample_cols].clip(lower=0) + 1.0)
                save_matrix_bundle(out_dir, "tcga", cancer, modality, matrix, features, samples)
                sample_frames.append(samples)
                feature_frames.append(features.assign(cohort="TCGA", cancer=cancer))
            except Exception as exc:
                logging.exception("Failed TCGA %s %s %s", cancer, modality, path)
                failures.append({"cohort": "TCGA", "cancer": cancer, "modality": modality, "path": str(path), "error": str(exc)})
        mut_path = project_root / "tcga-5" / TCGA_STUDIES[cancer] / TCGA_STUDIES[cancer] / "data_mutations.txt"
        if mut_path.exists():
            try:
                matrix, features, samples = derive_tcga_mutation_binary(mut_path, "TCGA", cancer, mapper)
                save_matrix_bundle(out_dir, "tcga", cancer, "mutation_binary", matrix, features, samples)
                sample_frames.append(samples)
                feature_frames.append(features.assign(cohort="TCGA", cancer=cancer))
            except Exception as exc:
                logging.exception("Failed TCGA mutation derivation %s", cancer)
                failures.append({"cohort": "TCGA", "cancer": cancer, "modality": "mutation_binary", "path": str(mut_path), "error": str(exc)})
    if sample_frames:
        pd.concat(sample_frames, ignore_index=True).drop_duplicates().to_parquet(out_dir / "clinical" / "sample_master.parquet", index=False)
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).drop_duplicates().to_parquet(out_dir / "features_manifest.parquet", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "qc" / "harmonize_failures.csv", index=False)


def read_tcga_clinical(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", na_values=NA_VALUES, low_memory=False)


def event_from_status(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).upper()
    if text.startswith("1:") or "DECEASED" in text or "PROGRESSION" in text or "RECUR" in text:
        return 1.0
    if text.startswith("0:") or "LIVING" in text or "CENSORED" in text or "ALIVE" in text:
        return 0.0
    return np.nan


def parse_clinical(project_root: Path, out_dir: Path) -> None:
    logging.info("Parsing clinical and target tables")
    patient_rows = []
    target_rows = []
    sample_rows = []
    for cancer in CANCERS:
        d = project_root / "cptac-5" / "downloads" / cancer
        if cancer != "ESCA":
            meta = d / f"{cancer}_meta.txt"
            surv = d / f"{cancer}_survival.txt"
            if meta.exists():
                df = pd.read_csv(meta, sep="\t", na_values=NA_VALUES)
                df = df[df["case_id"].astype(str) != "data_type"].copy()
                df["cohort"] = "CPTAC"
                df["cancer"] = cancer
                df["patient_id"] = [canonical_patient("CPTAC", cancer, x) for x in df["case_id"]]
                patient_rows.append(df)
            if surv.exists():
                sv = pd.read_csv(surv, sep="\t", na_values=NA_VALUES)
                sv["cohort"] = "CPTAC"
                sv["cancer"] = cancer
                sv["patient_id"] = [canonical_patient("CPTAC", cancer, x) for x in sv["case_id"]]
                sv["OS_days"] = pd.to_numeric(sv.get("OS_days"), errors="coerce")
                sv["OS_event"] = pd.to_numeric(sv.get("OS_event"), errors="coerce")
                sv["PFS_days"] = pd.to_numeric(sv.get("PFS_days"), errors="coerce")
                sv["PFS_event"] = pd.to_numeric(sv.get("PFS_event"), errors="coerce")
                target_rows.append(sv[["patient_id", "cohort", "cancer", "OS_days", "OS_event", "PFS_days", "PFS_event"]])
        else:
            for p in d.glob("*Clinical*Firehose.tsi"):
                raw = pd.read_csv(p, sep="\t", na_values=NA_VALUES)
                if "attrib_name" in raw.columns:
                    cli = raw.set_index("attrib_name").T.reset_index(names="case_id")
                    cli["cohort"] = "CPTAC"
                    cli["cancer"] = cancer
                    cli["source_note"] = "LinkedOmics_TCGA_ESCA_fallback"
                    cli["patient_id"] = [canonical_patient("CPTAC", cancer, x) for x in cli["case_id"]]
                    patient_rows.append(cli)
        study = TCGA_STUDIES[cancer]
        td = project_root / "tcga-5" / study / study
        patient_path = td / "data_clinical_patient.txt"
        sample_path = td / "data_clinical_sample.txt"
        if patient_path.exists():
            cli = read_tcga_clinical(patient_path)
            cli["cohort"] = "TCGA"
            cli["cancer"] = cancer
            cli["patient_id"] = [canonical_patient("TCGA", cancer, x) for x in cli["PATIENT_ID"]]
            patient_rows.append(cli)
            targets = pd.DataFrame({"patient_id": cli["patient_id"], "cohort": "TCGA", "cancer": cancer})
            for prefix in ["OS", "PFS", "DSS", "DFS"]:
                status = f"{prefix}_STATUS"
                months = f"{prefix}_MONTHS"
                if status in cli.columns:
                    targets[f"{prefix}_event"] = [event_from_status(x) for x in cli[status]]
                if months in cli.columns:
                    targets[f"{prefix}_days"] = pd.to_numeric(cli[months], errors="coerce") * 30.4375
            target_rows.append(targets)
        if sample_path.exists():
            sm = read_tcga_clinical(sample_path)
            sm["cohort"] = "TCGA"
            sm["cancer"] = cancer
            sm["sample_id"] = [canonical_sample("TCGA", cancer, x) for x in sm["SAMPLE_ID"]]
            sm["patient_id"] = [canonical_patient("TCGA", cancer, x) for x in sm["PATIENT_ID"]]
            sm["tissue"] = [tcga_tissue(x) for x in sm["SAMPLE_ID"]]
            sample_rows.append(sm)
    patient = pd.concat(patient_rows, ignore_index=True, sort=False) if patient_rows else pd.DataFrame()
    targets = pd.concat(target_rows, ignore_index=True, sort=False) if target_rows else pd.DataFrame()
    clinical_samples = pd.concat(sample_rows, ignore_index=True, sort=False) if sample_rows else pd.DataFrame()
    parquet_safe(patient).to_parquet(out_dir / "clinical" / "patient_master.parquet", index=False)
    parquet_safe(targets.drop_duplicates(subset=["patient_id"])).to_parquet(out_dir / "clinical" / "targets.parquet", index=False)
    parquet_safe(clinical_samples).to_parquet(out_dir / "clinical" / "tcga_clinical_samples.parquet", index=False)
    if (out_dir / "clinical" / "sample_master.parquet").exists() and not clinical_samples.empty:
        omic_samples = pd.read_parquet(out_dir / "clinical" / "sample_master.parquet")
        merged_samples = pd.concat([omic_samples, clinical_samples[omic_samples.columns.intersection(clinical_samples.columns)]], ignore_index=True, sort=False)
        parquet_safe(merged_samples.drop_duplicates()).to_parquet(out_dir / "clinical" / "sample_master.parquet", index=False)
    logging.info("Clinical patients: %d targets: %d", len(patient), len(targets))


def matrix_paths(out_dir: Path, cancer: str, modalities: Iterable[str]) -> list[tuple[str, str, Path]]:
    paths = []
    for cohort in ["cptac", "tcga"]:
        base = out_dir / "harmonized" / cohort / cancer
        for modality in modalities:
            p = base / f"{modality}.parquet"
            if p.exists():
                paths.append((cohort.upper(), modality, p))
    return paths


def modality_matrix_for_training(path: Path, cohort: str, modality: str, sample_master: pd.DataFrame, min_obs: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    if "sample_id" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    sample_info = sample_master[["sample_id", "patient_id", "cohort", "cancer", "tissue"]].drop_duplicates("sample_id")
    df = df.merge(sample_info, on="sample_id", how="left")
    df = df[df["tissue"].fillna("tumor").isin(["tumor", "unknown"])].copy()
    meta_cols = ["sample_id", "patient_id", "cohort", "cancer", "tissue"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    if not feature_cols:
        return pd.DataFrame(), pd.DataFrame()
    x = df.set_index("sample_id")[feature_cols].apply(pd.to_numeric, errors="coerce")
    obs = x.notna().mean(axis=0)
    x = x.loc[:, obs >= min_obs]
    nunique = x.nunique(dropna=True)
    x = x.loc[:, nunique > 1]
    x.columns = [f"{modality}::{c}" for c in x.columns]
    sample_meta = df.set_index("sample_id")[["patient_id", "cohort", "cancer", "tissue"]].drop_duplicates()
    return x, sample_meta


def normalize_part(x: pd.DataFrame, sample_meta: pd.DataFrame, modality: str, min_batch_obs: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = x.isna()
    if modality in BINARY_MODALITIES:
        return x.fillna(0), missing.astype(np.uint8)
    med = x.median(axis=0, skipna=True)
    z = x.fillna(med).astype(np.float32)
    # Conservative cohort centering for continuous shared features.
    if "cohort" in sample_meta.columns and sample_meta["cohort"].nunique(dropna=True) > 1:
        global_mean = z.mean(axis=0)
        for batch, idx in sample_meta.groupby("cohort").groups.items():
            if len(idx) >= min_batch_obs:
                cols = z.columns[z.loc[idx].notna().sum(axis=0) >= min_batch_obs]
                z.loc[idx, cols] = z.loc[idx, cols] - z.loc[idx, cols].mean(axis=0) + global_mean[cols]
    mean = z.mean(axis=0)
    std = z.std(axis=0).replace(0, 1.0).fillna(1.0)
    z = (z - mean) / std
    return z.fillna(0).astype(np.float32), missing.astype(np.uint8)


def audit_outliers_before_normalization(
    x: pd.DataFrame,
    sample_meta: pd.DataFrame,
    modality: str,
    out_dir: Path,
    cancer: str,
    view: str,
    cohort: str,
    z_threshold: float = 6.0,
    min_feature_obs: int = 10,
    max_cell_records: int = 100000,
) -> None:
    """Audit cell-level and sample-level outliers without modifying values."""
    if x.empty or modality not in OUTLIER_AUDIT_MODALITIES:
        return
    qc_dir = out_dir / "qc" / "outliers" / cancer / view
    qc_dir.mkdir(parents=True, exist_ok=True)
    numeric = x.apply(pd.to_numeric, errors="coerce")
    obs = numeric.notna().sum(axis=0)
    keep_cols = obs[obs >= min_feature_obs].index
    if len(keep_cols) == 0:
        return
    work = numeric[keep_cols]
    med = work.median(axis=0, skipna=True)
    mad = (work - med).abs().median(axis=0, skipna=True).replace(0, np.nan)
    modified_z = 0.6745 * (work - med) / mad
    mask = modified_z.abs() > z_threshold
    mask = mask.fillna(False)
    sample_counts = mask.sum(axis=1).astype(int)
    sample_summary = pd.DataFrame(
        {
            "sample_id": work.index,
            "patient_id": sample_meta.reindex(work.index).get("patient_id", pd.Series(index=work.index, dtype=object)).values,
            "cohort": cohort,
            "cancer": cancer,
            "view": view,
            "modality": modality,
            "outlier_cells": sample_counts.values,
            "audited_features": int(len(keep_cols)),
            "outlier_fraction": sample_counts.values / max(1, len(keep_cols)),
        }
    ).sort_values(["outlier_cells", "outlier_fraction"], ascending=False)
    sample_summary.to_csv(qc_dir / f"{cohort}_{modality}_sample_outliers.csv", index=False)

    feature_counts = mask.sum(axis=0).astype(int)
    feature_summary = pd.DataFrame(
        {
            "feature_id": feature_counts.index,
            "cohort": cohort,
            "cancer": cancer,
            "view": view,
            "modality": modality,
            "outlier_samples": feature_counts.values,
            "audited_samples": int(work.shape[0]),
            "outlier_fraction": feature_counts.values / max(1, work.shape[0]),
        }
    ).sort_values(["outlier_samples", "outlier_fraction"], ascending=False)
    feature_summary.to_csv(qc_dir / f"{cohort}_{modality}_feature_outliers.csv", index=False)

    coords = np.argwhere(mask.to_numpy())
    if coords.size:
        if len(coords) > max_cell_records:
            # Keep the most extreme cells when the audit is very large.
            z_values = np.abs(modified_z.to_numpy()[coords[:, 0], coords[:, 1]])
            keep = np.argsort(z_values)[-max_cell_records:]
            coords = coords[keep]
        rows = []
        index_values = work.index.to_numpy()
        columns = work.columns.to_numpy()
        values = work.to_numpy()
        z_arr = modified_z.to_numpy()
        for row_i, col_i in coords:
            rows.append(
                {
                    "sample_id": index_values[row_i],
                    "feature_id": columns[col_i],
                    "cohort": cohort,
                    "cancer": cancer,
                    "view": view,
                    "modality": modality,
                    "raw_value": float(values[row_i, col_i]),
                    "feature_median": float(med.iloc[col_i]),
                    "feature_mad": float(mad.iloc[col_i]),
                    "modified_z": float(z_arr[row_i, col_i]),
                }
            )
        pd.DataFrame(rows).sort_values("modified_z", key=lambda s: s.abs(), ascending=False).to_csv(qc_dir / f"{cohort}_{modality}_cell_outliers.csv", index=False)


def clinical_feature_matrix(out_dir: Path, cancer: str, sample_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_path = out_dir / "clinical" / "patient_master.parquet"
    if not patient_path.exists() or sample_meta.empty:
        return pd.DataFrame(index=sample_meta.index), pd.DataFrame(index=sample_meta.index)
    patient = pd.read_parquet(patient_path)
    patient = patient[patient["cancer"] == cancer].drop_duplicates("patient_id")
    exclude = {
        "OS_STATUS",
        "OS_MONTHS",
        "PFS_STATUS",
        "PFS_MONTHS",
        "DSS_STATUS",
        "DSS_MONTHS",
        "DFS_STATUS",
        "DFS_MONTHS",
        "OS_days",
        "OS_event",
        "PFS_days",
        "PFS_event",
        "case_id",
        "PATIENT_ID",
        "patient_id",
        "cohort",
        "cancer",
    }
    leakage_terms = [
        "NEW_TUMOR_EVENT",
        "PERSON_NEOPLASM_CANCER_STATUS",
        "TUMOR_STATUS",
        "PRIMARY_THERAPY_OUTCOME",
        "STATUS",
        "VITAL_STATUS",
    ]
    cli = sample_meta[["patient_id"]].merge(patient, on="patient_id", how="left")
    cli.index = sample_meta.index
    candidates = [
        c
        for c in cli.columns
        if c not in exclude and not any(term in str(c).upper() for term in leakage_terms)
    ]
    parts = []
    masks = []
    for col in candidates:
        s = cli[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.7:
            parts.append(num.rename(f"clinical::{col}"))
            masks.append(num.isna().astype(np.uint8).rename(f"clinical::{col}"))
        else:
            cat = s.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
            vc = cat.value_counts()
            keep = set(vc[vc >= max(5, int(0.01 * len(cat)))].index)
            cat = cat.where(cat.isin(keep), "Other")
            dummies = pd.get_dummies(cat, prefix=f"clinical::{col}", dtype=np.float32)
            parts.append(dummies)
            masks.append(pd.DataFrame(0, index=dummies.index, columns=dummies.columns, dtype=np.uint8))
    if not parts:
        return pd.DataFrame(index=sample_meta.index), pd.DataFrame(index=sample_meta.index)
    x = pd.concat(parts, axis=1)
    m = pd.concat(masks, axis=1)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(axis=0)).fillna(0).astype(np.float32)
    return x, m


def make_train_ready(
    project_root: Path,
    out_dir: Path,
    min_obs: float = 0.5,
    outlier_z: float = 6.0,
    outlier_min_feature_obs: int = 10,
    outlier_max_cell_records: int = 100000,
    cancers: list[str] | None = None,
) -> None:
    sample_master = pd.read_parquet(out_dir / "clinical" / "sample_master.parquet").drop_duplicates("sample_id")
    targets = pd.read_parquet(out_dir / "clinical" / "targets.parquet").drop_duplicates("patient_id")
    views = {"core": CORE_MODALITIES, "proteogenomic": PROTEOGENOMIC_MODALITIES}
    reports = []
    selected_cancers = cancers or CANCERS
    for cancer in selected_cancers:
        for view, modalities in views.items():
            logging.info("Creating train-ready view %s %s", cancer, view)
            part_frames = []
            mask_frames = []
            meta_frames = []
            for cohort, modality, path in matrix_paths(out_dir, cancer, modalities):
                x, smeta = modality_matrix_for_training(path, cohort, modality, sample_master, min_obs)
                if x.empty:
                    continue
                audit_outliers_before_normalization(
                    x,
                    smeta,
                    modality,
                    out_dir,
                    cancer,
                    view,
                    cohort,
                    z_threshold=outlier_z,
                    min_feature_obs=outlier_min_feature_obs,
                    max_cell_records=outlier_max_cell_records,
                )
                x_norm, m = normalize_part(x, smeta, modality)
                part_frames.append(x_norm)
                mask_frames.append(m)
                meta_frames.append(smeta)
                reports.append({"cancer": cancer, "view": view, "cohort": cohort, "modality": modality, "samples": x.shape[0], "features": x.shape[1]})
            if not part_frames:
                continue
            sample_meta = pd.concat(meta_frames).groupby(level=0).first()
            cli_x, cli_m = clinical_feature_matrix(out_dir, cancer, sample_meta)
            if not cli_x.empty:
                part_frames.append(cli_x)
                mask_frames.append(cli_m)
            all_samples = sorted(set().union(*[set(x.index) for x in part_frames]))
            aligned = [x.reindex(all_samples) for x in part_frames]
            aligned_masks = [m.reindex(all_samples).fillna(1).astype(np.uint8) for m in mask_frames]
            x_all = pd.concat(aligned, axis=1)
            miss = pd.concat(aligned_masks, axis=1)
            x_all = x_all.fillna(0).astype(np.float32)
            miss = miss.reindex(columns=x_all.columns).fillna(1).astype(np.uint8)
            sample_meta = sample_meta.reindex(all_samples)
            sample_meta = sample_meta.merge(targets, on="patient_id", how="left")
            has_target = sample_meta[["OS_days", "OS_event", "PFS_days", "PFS_event"]].notna().any(axis=1) if "OS_days" in sample_meta.columns else pd.Series(True, index=sample_meta.index)
            x_all = x_all.loc[has_target.values]
            miss = miss.loc[has_target.values]
            sample_meta = sample_meta.loc[has_target.values]
            dest = out_dir / "train_ready" / cancer / view
            dest.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(dest / "X.npz", sparse.csr_matrix(x_all.values.astype(np.float32)))
            sparse.save_npz(dest / "missing_mask.npz", sparse.csr_matrix(miss.values.astype(np.uint8)))
            pd.DataFrame({"sample_id": x_all.index}).join(sample_meta.reset_index(drop=True)).to_csv(dest / "sample_index.csv", index=False)
            pd.DataFrame({"feature_index": np.arange(x_all.shape[1]), "feature_id": x_all.columns}).to_csv(dest / "feature_index.csv", index=False)
            sample_ids = np.array(x_all.index)
            if len(sample_ids) >= 10:
                train, temp = train_test_split(sample_ids, test_size=0.30, random_state=42)
                val, test = train_test_split(temp, test_size=0.50, random_state=42)
            else:
                train, val, test = sample_ids, np.array([]), np.array([])
            (dest / "splits.json").write_text(json.dumps({"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}, indent=2))
            reports.append({"cancer": cancer, "view": view, "cohort": "ALL", "modality": "combined", "samples": x_all.shape[0], "features": x_all.shape[1]})
    report = pd.DataFrame(reports)
    report_path = out_dir / "qc" / "train_ready_report.csv"
    if cancers and report_path.exists():
        previous = pd.read_csv(report_path)
        previous = previous[~previous["cancer"].astype(str).isin(selected_cancers)]
        report = pd.concat([previous, report], ignore_index=True, sort=False)
    report.to_csv(report_path, index=False)


def gene_from_feature(feature_id: str) -> str | None:
    gene = None
    if "::GENE:" in feature_id:
        gene = feature_id.split("::GENE:", 1)[1]
    elif "::ENSG:" in feature_id:
        gene = "ENSG:" + feature_id.split("::ENSG:", 1)[1]
    elif "::ENTREZ:" in feature_id:
        gene = "ENTREZ:" + feature_id.split("::ENTREZ:", 1)[1]
    if gene is not None:
        return gene.split("|", 1)[0]
    return None


def modality_from_feature(feature_id: str) -> str:
    return feature_id.split("::", 1)[0]


def build_graph_tensors(out_dir: Path, graph_max_nodes: int = 2000, top_k: int = 5) -> None:
    for cancer in CANCERS:
        src = out_dir / "train_ready" / cancer / "proteogenomic"
        if not (src / "X.npz").exists():
            continue
        logging.info("Building graph tensors for %s", cancer)
        x_sparse = sparse.load_npz(src / "X.npz").tocsr()
        miss_sparse = sparse.load_npz(src / "missing_mask.npz").tocsr()
        features = pd.read_csv(src / "feature_index.csv")
        samples = pd.read_csv(src / "sample_index.csv")
        feature_ids = features["feature_id"].tolist()
        records = []
        for i, fid in enumerate(feature_ids):
            mod = modality_from_feature(fid)
            gene = gene_from_feature(fid)
            if gene and mod in GRAPH_MODALITIES:
                records.append((i, mod, gene, fid))
        if not records:
            continue
        rec = pd.DataFrame(records, columns=["col", "modality", "gene", "feature_id"])
        activity_cols = rec[rec["modality"].eq("rnaseq_gene")]["col"].to_numpy()
        if len(activity_cols) == 0:
            activity_cols = rec["col"].drop_duplicates().to_numpy()
        activity = np.asarray(x_sparse[:, activity_cols].todense(), dtype=np.float32)
        genes_for_activity = rec.set_index("col").loc[activity_cols, "gene"].to_numpy()
        gene_scores = pd.DataFrame(activity, columns=genes_for_activity).groupby(level=0, axis=1).mean()
        variances = gene_scores.var(axis=0).sort_values(ascending=False)
        nodes = variances.index[:graph_max_nodes].tolist()
        node_index = {g: i for i, g in enumerate(nodes)}
        node_rec = rec[rec["gene"].isin(node_index)].copy()
        node_rec["node"] = node_rec["gene"].map(node_index)
        modalities = [m for m in GRAPH_MODALITIES if m in set(node_rec["modality"])]
        mod_index = {m: i for i, m in enumerate(modalities)}
        edge_index, edge_attr = correlation_edges(gene_scores[nodes].to_numpy(dtype=np.float32), top_k=top_k)
        graphs = {}
        for row_i, sample_id in enumerate(samples["sample_id"]):
            vals = np.zeros((len(nodes), len(modalities)), dtype=np.float32)
            masks = np.ones((len(nodes), len(modalities)), dtype=np.float32)
            row_values = x_sparse.getrow(row_i)
            row_missing = miss_sparse.getrow(row_i)
            for _, r in node_rec.iterrows():
                node = int(r["node"])
                mod = mod_index[r["modality"]]
                col = int(r["col"])
                vals[node, mod] = row_values[0, col]
                masks[node, mod] = 1.0 - row_missing[0, col]
            y = {}
            for col in ["OS_days", "OS_event", "PFS_days", "PFS_event"]:
                if col in samples.columns and pd.notna(samples.loc[row_i, col]):
                    y[col] = float(samples.loc[row_i, col])
            graphs[sample_id] = {
                "x": torch.tensor(np.concatenate([vals, masks], axis=1), dtype=torch.float32),
                "edge_index": torch.tensor(edge_index, dtype=torch.long),
                "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
                "y": y,
            }
        dest = out_dir / "graphs" / cancer
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, dest / "graph_tensors.pt")
        pd.DataFrame({"node_index": range(len(nodes)), "gene": nodes}).to_csv(dest / "nodes.csv", index=False)
        pd.DataFrame({"modality_index": range(len(modalities)), "modality": modalities}).to_csv(dest / "modalities.csv", index=False)
        logging.info("Saved %d graphs for %s", len(graphs), cancer)


def correlation_edges(x: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[1] <= 1:
        return np.array([[0], [0]], dtype=np.int64), np.array([[1.0]], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    edges = []
    attrs = []
    k = min(top_k, corr.shape[0] - 1)
    for i in range(corr.shape[0]):
        nbrs = np.argpartition(np.abs(corr[i]), -k)[-k:]
        for j in nbrs:
            if i != j:
                edges.append((i, int(j)))
                attrs.append([float(corr[i, j])])
    if not edges:
        edges = [(0, 0)]
        attrs = [[1.0]]
    return np.array(edges, dtype=np.int64).T, np.array(attrs, dtype=np.float32)


def validate_outputs(out_dir: Path) -> None:
    logging.info("Validating outputs")
    rows = []
    targets = out_dir / "clinical" / "targets.parquet"
    if targets.exists():
        t = pd.read_parquet(targets)
        rows.append({"check": "targets_rows", "value": len(t), "status": "ok" if len(t) else "fail"})
        rows.append({"check": "unique_target_patients", "value": t["patient_id"].nunique(), "status": "ok"})
    for path in sorted((out_dir / "train_ready").glob("*/*/X.npz")):
        view_dir = path.parent
        x = sparse.load_npz(path)
        samples = pd.read_csv(view_dir / "sample_index.csv")
        features = pd.read_csv(view_dir / "feature_index.csv")
        rows.append({"check": f"{view_dir.relative_to(out_dir)}_shape", "value": f"{x.shape[0]}x{x.shape[1]}", "status": "ok" if x.shape == (len(samples), len(features)) else "fail"})
        leakage = [
            f
            for f in features["feature_id"]
            if str(f).startswith("clinical::")
            and re.search(r"(OS_|PFS_|DSS_|DFS_|STATUS|MONTHS|EVENT|TUMOR_STATUS|VITAL_STATUS)", str(f), re.I)
        ]
        rows.append({"check": f"{view_dir.relative_to(out_dir)}_leakage_features", "value": len(leakage), "status": "ok" if len(leakage) == 0 else "warn"})
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "qc" / "validation_report.csv", index=False)
    logging.info("Validation checks: %d", len(report))


def run_all(args: argparse.Namespace) -> None:
    ensure_dirs(args.out_dir)
    build_inventory(args.project_root, args.out_dir)
    build_gene_reference(args.project_root, args.out_dir, download=not args.no_download_reference)
    harmonize_omics(args.project_root, args.out_dir)
    parse_clinical(args.project_root, args.out_dir)
    make_train_ready(
        args.project_root,
        args.out_dir,
        min_obs=args.min_obs,
        outlier_z=args.outlier_z,
        outlier_min_feature_obs=args.outlier_min_feature_obs,
        outlier_max_cell_records=args.outlier_max_cell_records,
        cancers=args.cancers,
    )
    build_graph_tensors(args.out_dir, graph_max_nodes=args.graph_max_nodes, top_k=args.graph_top_k)
    validate_outputs(args.out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "inventory", "reference", "harmonize", "clinical", "train-ready", "graphs", "validate"])
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("TABR1_DATA_ROOT", "data/raw")))
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get("TABR1_PROCESSED_ROOT", "data/processed")))
    parser.add_argument("--min-obs", type=float, default=0.50)
    parser.add_argument("--graph-max-nodes", type=int, default=2000)
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument("--outlier-z", type=float, default=6.0)
    parser.add_argument("--outlier-min-feature-obs", type=int, default=10)
    parser.add_argument("--outlier-max-cell-records", type=int, default=100000)
    parser.add_argument("--cancers", default=None, help="Comma-separated cancer list for train-ready/all; default uses all cancers.")
    parser.add_argument("--no-download-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.cancers = [c.strip().upper() for c in args.cancers.split(",") if c.strip()] if args.cancers else None
    ensure_dirs(args.out_dir)
    if args.command == "all":
        run_all(args)
    elif args.command == "inventory":
        build_inventory(args.project_root, args.out_dir)
    elif args.command == "reference":
        build_gene_reference(args.project_root, args.out_dir, download=not args.no_download_reference)
    elif args.command == "harmonize":
        harmonize_omics(args.project_root, args.out_dir)
    elif args.command == "clinical":
        parse_clinical(args.project_root, args.out_dir)
    elif args.command == "train-ready":
        make_train_ready(
            args.project_root,
            args.out_dir,
            min_obs=args.min_obs,
            outlier_z=args.outlier_z,
            outlier_min_feature_obs=args.outlier_min_feature_obs,
            outlier_max_cell_records=args.outlier_max_cell_records,
            cancers=args.cancers,
        )
    elif args.command == "graphs":
        build_graph_tensors(args.out_dir, graph_max_nodes=args.graph_max_nodes, top_k=args.graph_top_k)
    elif args.command == "validate":
        validate_outputs(args.out_dir)


if __name__ == "__main__":
    main()

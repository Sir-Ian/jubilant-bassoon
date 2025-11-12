"""
BA Product Advisor — Dataset Merge Script
-----------------------------------------
Transforms the provided beauty store and retail basket datasets into a synthetic
beauty-focused transaction dataset that can power downstream recommendation
prototypes.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SKINCARE_PATH = BASE_DIR / "Global_skincare_ Beauty_store.csv"
RETAIL_PATH = BASE_DIR / "Retail_Transactions_Dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "merged_beauty_transactions.csv"

# Distributions used for synthetic demographic attributes
AGE_GROUPS = ["18-25", "26-40", "41-55", "55+"]
AGE_WEIGHTS = [0.25, 0.4, 0.25, 0.1]
SKIN_TYPES = ["Dry", "Oily", "Combination", "Sensitive"]
SKIN_TYPE_WEIGHTS = [0.35, 0.2, 0.3, 0.15]

# Lightweight heuristics that bias the product sampling step
SEASON_KEYWORDS = {
    "winter": ["cream", "moisturizer", "butter", "balm"],
    "spring": ["serum", "brightening", "mist"],
    "summer": ["sunscreen", "spf", "gel"],
    "fall": ["oil", "mask", "lotion"],
}

CUSTOMER_CATEGORY_KEYWORDS = {
    "student": ["acne", "cleanser", "wash"],
    "athlete": ["deodorant", "body", "cool"],
    "homemaker": ["lotion", "body"],
    "business": ["serum", "anti-aging"],
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names to snake_case."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def load_data(skincare_path: Path, retail_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV inputs."""
    if not skincare_path.exists():
        raise FileNotFoundError(f"Could not find skincare dataset at {skincare_path}")
    if not retail_path.exists():
        raise FileNotFoundError(f"Could not find retail dataset at {retail_path}")

    skincare_df = pd.read_csv(skincare_path)
    retail_df = pd.read_csv(retail_path)
    return skincare_df, retail_df


def parse_product_list(value) -> List[str]:
    """Turn a serialized product list into a Python list."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = []
    for part in text.split(","):
        cleaned = part.strip().strip("'\"")
        if cleaned:
            parts.append(cleaned)
    return parts or [str(value).strip()]


def preprocess_skincare(skincare_df: pd.DataFrame) -> pd.DataFrame:
    """Clean skincare dataset columns and important data types."""
    skincare_df = clean_column_names(skincare_df.copy())
    if "order_date" in skincare_df.columns:
        skincare_df["order_date"] = pd.to_datetime(
            skincare_df["order_date"], format="%m/%d/%y", errors="coerce"
        )
    for col in ["city", "state", "region", "country", "market"]:
        if col in skincare_df.columns:
            skincare_df[col] = (
                skincare_df[col].astype(str).str.strip().str.title().replace("Nan", np.nan)
            )
    numeric_cols = {"quantity", "sales", "discount", "profit"}
    for col in numeric_cols.intersection(skincare_df.columns):
        skincare_df[col] = pd.to_numeric(skincare_df[col], errors="coerce")
    return skincare_df


def preprocess_retail(retail_df: pd.DataFrame) -> pd.DataFrame:
    """Clean retail dataset columns and convert fields needed for enrichment."""
    retail_df = clean_column_names(retail_df.copy())
    if "date" in retail_df.columns:
        retail_df["date"] = pd.to_datetime(retail_df["date"], errors="coerce")
    if "product" in retail_df.columns:
        retail_df["product_list"] = retail_df["product"].apply(parse_product_list)
    if "city" in retail_df.columns:
        retail_df["city"] = retail_df["city"].astype(str).str.strip().str.title()
    if "total_items" in retail_df.columns:
        retail_df["total_items"] = (
            pd.to_numeric(retail_df["total_items"], errors="coerce").fillna(1).astype(int)
        )
    if "total_cost" in retail_df.columns:
        retail_df["total_cost"] = pd.to_numeric(retail_df["total_cost"], errors="coerce")
    if "discount_applied" in retail_df.columns:
        retail_df["discount_applied"] = retail_df["discount_applied"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    if "season" in retail_df.columns:
        retail_df["season"] = retail_df["season"].astype(str).str.strip().str.title()
    if "customer_category" in retail_df.columns:
        retail_df["customer_category"] = (
            retail_df["customer_category"].astype(str).str.strip().str.title()
        )
    return retail_df


def build_beauty_catalog(skincare_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unique beauty products with representative metrics."""
    required_cols = {"product", "category", "subcategory"}
    missing = required_cols - set(skincare_df.columns)
    if missing:
        raise KeyError(f"Skincare dataset missing required columns: {missing}")

    catalog = (
        skincare_df.dropna(subset=["product"])
        .groupby(["product", "category", "subcategory"], dropna=False)
        .agg(avg_sales=("sales", "mean"), avg_profit=("profit", "mean"), avg_discount=("discount", "mean"))
        .reset_index()
    )
    catalog["category"] = catalog["category"].fillna("Unknown")
    catalog["subcategory"] = catalog["subcategory"].fillna("Unknown")
    catalog[["avg_sales", "avg_profit", "avg_discount"]] = catalog[
        ["avg_sales", "avg_profit", "avg_discount"]
    ].fillna(0)
    return catalog


class CatalogSampler:
    """Cache-aware helper that returns catalog subsets for keyword combinations."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog
        self._full_index = catalog.index.to_numpy()
        self._cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self._product_lower = catalog["product"].fillna("").str.lower()
        self._subcategory_lower = catalog["subcategory"].fillna("").str.lower()

    @staticmethod
    def _normalize_keywords(keywords: List[str]) -> Tuple[str, ...]:
        normalized = sorted({kw.strip().lower() for kw in keywords if kw})
        return tuple(normalized)

    def _indices_for_keywords(self, keywords: List[str]) -> np.ndarray:
        key = self._normalize_keywords(keywords)
        if key in self._cache:
            return self._cache[key]

        if not key:
            indices = self._full_index
        else:
            mask = np.zeros(len(self.catalog), dtype=bool)
            for kw in key:
                contains_product = self._product_lower.str.contains(kw, na=False)
                contains_subcat = self._subcategory_lower.str.contains(kw, na=False)
                mask |= (contains_product | contains_subcat).to_numpy()
            indices = self.catalog.index[mask].to_numpy()
            if indices.size == 0:
                indices = self._full_index

        self._cache[key] = indices
        return indices

    def sample_indices(
        self, keywords: List[str], total_items: int, rng: np.random.Generator
    ) -> np.ndarray:
        indices = self._indices_for_keywords(keywords)
        replace = indices.size < total_items
        chosen = rng.choice(indices, size=total_items, replace=replace)
        return chosen.astype(int, copy=False)

    def select(self, keywords: List[str], total_items: int, rng: np.random.Generator) -> pd.DataFrame:
        chosen = self.sample_indices(keywords, total_items, rng)
        return self.catalog.loc[chosen].reset_index(drop=True)


def build_city_context(skincare_df: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
    """Create a lookup of city -> location metadata + customer list for sampling."""
    context: Dict[str, Dict[str, Optional[str]]] = {}
    for _, row in skincare_df.iterrows():
        city = row.get("city")
        if pd.isna(city):
            continue
        key = str(city).strip().title()
        city_entry = context.setdefault(
            key,
            {
                "region": row.get("region"),
                "state": row.get("state"),
                "country": row.get("country"),
                "customer_ids": [],
            },
        )
        customer_id = row.get("customer_id")
        if customer_id and customer_id not in city_entry["customer_ids"]:
            city_entry["customer_ids"].append(customer_id)
    return context


def select_beauty_products(
    season: str,
    customer_category: str,
    sampler: CatalogSampler,
    total_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample beauty products with a mild bias from season/customer segments."""
    season_key = str(season or "").strip().lower()
    category_key = str(customer_category or "").strip().lower()
    keywords = SEASON_KEYWORDS.get(season_key, []) + CUSTOMER_CATEGORY_KEYWORDS.get(category_key, [])
    return sampler.sample_indices(keywords, total_items, rng)


def derive_customer_id(
    city: str,
    city_context: Dict[str, Dict[str, Optional[str]]],
    fallback_ids: List[str],
    rng: np.random.Generator,
) -> str:
    """Pick an existing customer if available, otherwise synthesize one."""
    context = city_context.get(city)
    if context and context["customer_ids"]:
        return str(rng.choice(context["customer_ids"]))
    if fallback_ids:
        return str(rng.choice(fallback_ids))
    return f"BA-CUST-{rng.integers(1000, 9999)}"


def _generate_chunk_dataframe(
    retail_chunk: pd.DataFrame,
    catalog: pd.DataFrame,
    city_context: Dict[str, Dict[str, Optional[str]]],
    fallback_ids: List[str],
    seed: int,
) -> pd.DataFrame:
    """Process a retail subset and return the enriched synthetic rows."""
    rng = np.random.default_rng(seed=seed)
    sampler = CatalogSampler(catalog)
    product_names_arr = catalog["product"].fillna("").astype(str).to_numpy()
    category_arr = catalog["category"].fillna("Unknown").astype(str).to_numpy()
    subcategory_arr = catalog["subcategory"].fillna("Unknown").astype(str).to_numpy()
    avg_sales_arr = pd.to_numeric(catalog["avg_sales"], errors="coerce").fillna(0).to_numpy()
    avg_profit_arr = pd.to_numeric(catalog["avg_profit"], errors="coerce").fillna(0).to_numpy()

    def series_or_default(df: pd.DataFrame, column: str, default_value):
        if column in df.columns:
            return df[column]
        return pd.Series([default_value] * len(df))

    city_series = (
        series_or_default(retail_chunk, "city", "Unknown")
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
        .str.title()
    )
    season_series = (
        series_or_default(retail_chunk, "season", "")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )
    category_series = (
        series_or_default(retail_chunk, "customer_category", "")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )
    transaction_ids = series_or_default(retail_chunk, "transaction_id", None)
    date_series = series_or_default(retail_chunk, "date", pd.NaT)
    total_cost_series = pd.to_numeric(
        series_or_default(retail_chunk, "total_cost", np.nan), errors="coerce"
    )
    total_items_series = (
        pd.to_numeric(series_or_default(retail_chunk, "total_items", 1), errors="coerce")
        .fillna(1)
        .astype(int)
    ).clip(lower=1)
    discount_series = (
        series_or_default(retail_chunk, "discount_applied", False).fillna(False).astype(bool)
    )

    columns = {
        "Transaction_ID": [],
        "Date": [],
        "Customer_ID": [],
        "City": [],
        "Region": [],
        "Country": [],
        "Customer_Category": [],
        "Age_Group": [],
        "Skin_Type": [],
        "Products": [],
        "Total_Items": [],
        "Total_Cost": [],
        "Category": [],
        "Subcategory": [],
        "Sales": [],
        "Discount": [],
        "Profit": [],
        "Season": [],
    }

    for idx, (
        transaction_id,
        txn_date,
        city,
        season,
        customer_category,
        total_cost,
        total_items,
        discount_flag,
    ) in enumerate(
        zip(
            transaction_ids,
            date_series,
            city_series,
            season_series,
            category_series,
            total_cost_series,
            total_items_series,
            discount_series,
        ),
        start=1,
    ):
        sampled_indices = select_beauty_products(
            season, customer_category, sampler, int(total_items), rng
        )
        selected_sales = float(avg_sales_arr[sampled_indices].sum())
        selected_profit = float(avg_profit_arr[sampled_indices].sum())

        discount_rate = float(rng.uniform(0.05, 0.25)) if discount_flag else 0.0
        discount_value = round(selected_sales * discount_rate, 2)
        adjusted_sales = round(max(selected_sales - discount_value, 0), 2)
        adjusted_profit = round(max(selected_profit - (discount_value * 0.5), 0), 2)

        product_names = product_names_arr[sampled_indices].tolist()
        category_labels = sorted(set(category_arr[sampled_indices].tolist()))
        subcategory_labels = sorted(set(subcategory_arr[sampled_indices].tolist()))

        age_group = str(rng.choice(AGE_GROUPS, p=AGE_WEIGHTS))
        skin_type = str(rng.choice(SKIN_TYPES, p=SKIN_TYPE_WEIGHTS))

        lookup_city = city if city else "Unknown"
        location_meta = city_context.get(lookup_city, {})
        customer_id = derive_customer_id(lookup_city, city_context, fallback_ids, rng)

        columns["Transaction_ID"].append(transaction_id)
        columns["Date"].append(txn_date)
        columns["Customer_ID"].append(customer_id)
        columns["City"].append(lookup_city)
        columns["Region"].append(location_meta.get("region", "Unknown"))
        columns["Country"].append(location_meta.get("country", "Unknown"))
        columns["Customer_Category"].append(customer_category)
        columns["Age_Group"].append(age_group)
        columns["Skin_Type"].append(skin_type)
        columns["Products"].append(json.dumps(product_names))
        columns["Total_Items"].append(int(total_items))
        columns["Total_Cost"].append(total_cost)
        columns["Category"].append("; ".join(category_labels))
        columns["Subcategory"].append("; ".join(subcategory_labels))
        columns["Sales"].append(adjusted_sales)
        columns["Discount"].append(discount_value)
        columns["Profit"].append(adjusted_profit)
        columns["Season"].append(season)

        if idx % 50000 == 0:
            print(f"Chunk processed {idx} rows...", flush=True)

    return pd.DataFrame(columns)


def merge_datasets(
    retail_df: pd.DataFrame,
    catalog: pd.DataFrame,
    city_context: Dict[str, Dict[str, Optional[str]]],
    skincare_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create the synthetic, beauty-focused transaction dataset."""
    row_count = len(retail_df)
    fallback_ids = skincare_df["customer_id"].dropna().astype(str).tolist()

    if row_count == 0:
        return pd.DataFrame()

    # Large datasets are processed in parallel chunks to keep runtime manageable.
    parallel_threshold = 200_000
    if row_count <= parallel_threshold:
        merged_df = _generate_chunk_dataframe(
            retail_df, catalog, city_context, fallback_ids, seed=42
        )
    else:
        cpu_count = max(2, (os.cpu_count() or 2) - 1)
        index_chunks = np.array_split(np.arange(row_count), cpu_count)
        chunks = [retail_df.iloc[idxs].copy() for idxs in index_chunks if len(idxs) > 0]
        results = []
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = []
            for idx, chunk in enumerate(chunks):
                seed = 42 + idx
                futures.append(
                    executor.submit(
                        _generate_chunk_dataframe,
                        chunk,
                        catalog,
                        city_context,
                        fallback_ids,
                        seed,
                    )
                )
            for idx, future in enumerate(as_completed(futures), start=1):
                chunk_df = future.result()
                results.append(chunk_df)
                print(
                    f"Completed chunk {idx}/{len(futures)} ({len(chunk_df)} rows)",
                    flush=True,
                )
        merged_df = pd.concat(results, ignore_index=True)

    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    return merged_df


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the merged dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def simulate_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create lightweight sanity-check plots if matplotlib is available."""
    # TODO: Expand visualization layer with richer QA dashboards for ML prep.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping visualization step.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    by_season = df.groupby("Season")["Total_Items"].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    by_season.plot(kind="bar", ax=ax, color="#FF7F50")
    ax.set_title("Beauty Items Per Season")
    ax.set_ylabel("Total Items")
    ax.set_xlabel("Season")
    plt.tight_layout()
    plot_path = output_dir / "beauty_items_by_season.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}", flush=True)


def transform_data(skincare_path: Path, retail_path: Path) -> pd.DataFrame:
    """Full end-to-end transformation pipeline."""
    start = time.time()
    skincare_df, retail_df = load_data(skincare_path, retail_path)
    print(f"Loaded datasets in {time.time() - start:.2f}s", flush=True)

    stage = time.time()
    skincare_df = preprocess_skincare(skincare_df)
    print(f"Preprocessed skincare data in {time.time() - stage:.2f}s", flush=True)

    stage = time.time()
    retail_df = preprocess_retail(retail_df)
    print(f"Preprocessed retail data in {time.time() - stage:.2f}s", flush=True)

    stage = time.time()
    catalog = build_beauty_catalog(skincare_df)
    city_context = build_city_context(skincare_df)
    print(f"Built catalog/context in {time.time() - stage:.2f}s", flush=True)

    stage = time.time()
    merged_df = merge_datasets(retail_df, catalog, city_context, skincare_df)
    print(f"Merged datasets in {time.time() - stage:.2f}s", flush=True)
    return merged_df


def main() -> None:
    """Script entry-point."""
    merged_df = transform_data(SKINCARE_PATH, RETAIL_PATH)
    save_output(merged_df, OUTPUT_PATH)

    # TODO: Add feature engineering artifacts (scalers, encoders) for model training.
    # TODO: Persist train/validation splits and candidate recommendation labels.

    print("Merged dataset preview:", flush=True)
    print(merged_df.head().to_string(index=False), flush=True)

    # Optional sanity-check visualization
    simulate_visualizations(merged_df, OUTPUT_PATH.parent)


if __name__ == "__main__":
    main()

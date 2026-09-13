import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
 
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

# 1. LOAD
 
def load_data(path: str) -> pd.DataFrame:
    """Load the raw restaurant workbook exactly as provided."""
    df = pd.read_excel(path)
    return df

2. EXPLORATORY DATA ANALYSIS

def run_eda(df: pd.DataFrame) -> dict:
    """Collect EDA facts used in the report. Returns a dict of computed
    values so the written report can quote exact, reproducible numbers."""
    facts = {}
    facts["shape"] = df.shape
    facts["dtypes"] = df.dtypes.astype(str).to_dict()
    facts["missing"] = df.isna().sum()
    facts["missing"] = facts["missing"][facts["missing"] > 0]
    facts["exact_duplicates"] = int(df.duplicated().sum())
    facts["duplicate_ids"] = int(df["Restaurant ID"].duplicated().sum())
    facts["n_cities"] = df["City"].nunique()
    facts["top_cities"] = df["City"].value_counts().head(10)
    facts["currency_counts"] = df["Currency"].value_counts()
    facts["price_range_counts"] = df["Price range"].value_counts().sort_index()
    facts["rating_text_counts"] = df["Rating text"].value_counts()
    facts["zero_rating_count"] = int((df["Aggregate rating"] == 0).sum())
    facts["zero_rating_with_votes"] = int(
        ((df["Aggregate rating"] == 0) & (df["Votes"] > 0)).sum()
    )
    facts["zero_cost_count"] = int((df["Average Cost for two"] == 0).sum())
    facts["switch_menu_counts"] = df["Switch to order menu"].value_counts()
    facts["is_delivering_counts"] = df["Is delivering now"].value_counts()
    facts["table_booking_counts"] = df["Has Table booking"].value_counts()
    facts["online_delivery_counts"] = df["Has Online delivery"].value_counts()
    cu = df["Cuisines"].dropna().str.split(",").explode().str.strip()
    facts["n_unique_cuisines"] = cu.nunique()
    facts["top_cuisines"] = cu.value_counts().head(15)
    return facts

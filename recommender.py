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

    # 3. PREPROCESSING
 
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing decisions (each explained):
 
    1. Cuisines missing (9 rows, 0.09%) -> DROPPED. Cuisine is the single
       most important content-based signal in this project; there is no
       reliable, non-fabricated way to impute a restaurant's cuisine, and
       0.09% row loss has no meaningful effect on the model.
 
    2. 'Switch to order menu' has a single constant value ('No') across all
       9,551 rows -> DROPPED. Zero variance = zero information for
       similarity computation.
 
    3. 'Rating color' and 'Rating text' are deterministic, one-to-one
       derivatives of 'Aggregate rating' (Zomato computes the color/text
       label directly from the numeric score) -> EXCLUDED from the
       similarity feature vector to avoid double-counting the same signal.
       'Rating text' is still kept in the dataframe for human-readable
       display in results tables.
 
    4. 'Is delivering now' is almost constant (99.6% 'No') -> excluded from
       the similarity vector as a near-zero-variance feature; kept for
       display only.
 
    5. Rows with Aggregate rating == 0 all carry Rating text == 'Not rated'
       (2,148 rows / 22.5%). Critically, ~1,054 of these already have 1-3
       votes -- i.e. "0.0" here means "not enough votes yet to display a
       score" and is a data-availability flag, NOT a genuine 1-star-style
       rating. We therefore add an explicit `Is Rated` boolean column so the
       recommender can treat "not yet rated" differently from "poorly
       rated" instead of silently treating both as the worst possible score.
 
    6. 'Average Cost for two' is NOT used as a similarity feature. The
       dataset mixes 12 different currencies (Indian Rupees, US Dollars,
       Pounds, Brazilian Real, etc.) and a raw numeric cost is not
       comparable across currencies (e.g. 400 INR and 400 IDR are wildly
       different amounts). 'Price range' (1-4) is Zomato's own
       currency-normalized affordability tier and IS used instead. Raw
       cost + currency are kept only for display purposes.
 
    7. 'Country Code' is a bare integer in this file with no accompanying
       decode table (no country-name mapping sheet was provided) -> kept
       as an opaque id for potential exact-match filtering only; 'City' is
       used as the primary, human-readable location signal instead. This
       is stated explicitly rather than guessing country names.
 
    8. A handful of city names (54 rows) contain corrupted/mojibake
       characters (e.g. "Bras..lia", "S..o Paulo") from an encoding issue
       in the source file. These are left completely UNCHANGED in the
       data used for computation (renaming them would be inventing text
       not verifiably present in the file); it is only noted as a known
       data-quality issue. Because the corruption is consistent per city,
       exact-match location filtering is unaffected.
 
    9. No exact duplicate rows and no duplicate Restaurant IDs were found,
       so no de-duplication step is required.
    """
    data = df.copy()
 
    # (1) drop rows with missing cuisine
    dropped_missing_cuisine = int(data["Cuisines"].isna().sum())
    data = data.dropna(subset=["Cuisines"]).reset_index(drop=True)
 
    # (5) explicit "was this restaurant actually rated" flag
    data["Is Rated"] = data["Aggregate rating"] > 0
 
    # tidy cuisine strings into clean lists, e.g. "Cafe, Tea" -> ["Cafe","Tea"]
    data["Cuisine List"] = (
        data["Cuisines"].str.split(",").apply(lambda lst: [c.strip() for c in lst])
    )
 
    # binary service-style features (kept: real variance, real signal)
    data["Table Booking Bin"] = (data["Has Table booking"] == "Yes").astype(int)
    data["Online Delivery Bin"] = (data["Has Online delivery"] == "Yes").astype(int)
 
    # scaled numeric features -> [0, 1]
    data["Price Scaled"] = (data["Price range"] - 1) / 3.0          # 1..4 -> 0..1
    data["Rating Scaled"] = data["Aggregate rating"] / 5.0           # 0..5 -> 0..1
 
    meta = {"dropped_missing_cuisine": dropped_missing_cuisine}
    return data, meta
 

# 4. FEATURE ENGINEERING / CONTENT-BASED REPRESENTATION
 
class RestaurantFeatureSpace:
    """
    Builds one consistent vector space that both restaurants and user
    preference profiles are projected into, so cosine similarity between
    the two is meaningful.
 
    Vector layout (concatenated, each block optionally weighted):
        [ cuisine multi-hot (n_cuisines dims) * w_cuisine,
          price_scaled (1 dim)               * w_price,
          rating_scaled (1 dim)               * w_rating,
          table_booking_bin (1 dim)           * w_service,
          online_delivery_bin (1 dim)         * w_service ]
 
    Cuisine gets a multi-hot (not one-hot) representation because a
    restaurant can legitimately serve several cuisines at once
    (e.g. "French, Japanese, Desserts").
    """
 
    DEFAULT_WEIGHTS = {"cuisine": 3.0, "price": 1.0, "rating": 1.0, "service": 0.5}
 
    def __init__(self, data: pd.DataFrame):
        self.mlb = MultiLabelBinarizer()
        self.cuisine_matrix = self.mlb.fit_transform(data["Cuisine List"])
        self.cuisine_classes = list(self.mlb.classes_)
        self.data = data
 
    def build_matrix(self, weights: dict = None) -> np.ndarray:
        w = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        blocks = [
            self.cuisine_matrix * w["cuisine"],
            (self.data["Price Scaled"].to_numpy().reshape(-1, 1)) * w["price"],
            (self.data["Rating Scaled"].to_numpy().reshape(-1, 1)) * w["rating"],
            (self.data["Table Booking Bin"].to_numpy().reshape(-1, 1)) * w["service"],
            (self.data["Online Delivery Bin"].to_numpy().reshape(-1, 1)) * w["service"],
        ]
        return np.hstack(blocks).astype(float)
 
    def build_user_vector(self, profile: dict, weights: dict = None) -> np.ndarray:
        w = {**self.DEFAULT_WEIGHTS, **(weights or {})}
 
        cuisine_vec = np.zeros(len(self.cuisine_classes))
        for c in profile.get("cuisines", []) or []:
            if c in self.cuisine_classes:
                cuisine_vec[self.cuisine_classes.index(c)] = 1.0
            else:
                raise ValueError(
                    f"Cuisine '{c}' not found in dataset's known cuisine list."
                )
        cuisine_vec *= w["cuisine"]
 
        # if the user has no price/rating opinion, use a neutral midpoint
        # (0.5) so that dimension contributes no directional bias
        price_target = profile.get("price_range")
        price_val = ((price_target - 1) / 3.0) if price_target else 0.5
        price_val *= w["price"]
 
        rating_target = profile.get("min_rating")
        rating_val = (rating_target / 5.0) if rating_target else 0.5
        rating_val *= w["rating"]
 
        table_val = w["service"] if profile.get("prefers_table_booking") else 0.0
        delivery_val = w["service"] if profile.get("prefers_online_delivery") else 0.0
 
        return np.concatenate(
            [cuisine_vec, [price_val], [rating_val], [table_val], [delivery_val]]
        )
 
 

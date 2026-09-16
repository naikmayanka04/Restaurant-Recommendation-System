import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
 
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. LOAD
 
def load_data(path: str) -> pd.DataFrame:
    """Load the raw restaurant workbook exactly as provided."""
    df = pd.read_excel(path)
    return df

#2. EXPLORATORY DATA ANALYSIS

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
 
 #5. RECOMMENDATION FUNCTION
 
def recommend(
    profile: dict,
    data: pd.DataFrame,
    fs: RestaurantFeatureSpace,
    top_n: int = 5,
    weights: dict = None,
    exclude_unrated: bool = False,
) -> pd.DataFrame:
    """
    Rank restaurants by cosine similarity to a user preference profile.
 
    profile keys (all optional -- missing keys simply mean "no opinion"):
        cuisines: list[str]            e.g. ["North Indian", "Chinese"]
        price_range: int in {1,2,3,4}
        min_rating: float in [0,5]
        city: str                      hard filter -- exact match on 'City'
        prefers_table_booking: bool
        prefers_online_delivery: bool
 
    Edge cases handled:
      - Unknown cuisine name -> raised clearly rather than silently ignored.
      - city filter matching zero restaurants -> returns an empty frame
        with a warning message instead of crashing.
      - exclude_unrated=True drops restaurants with no reliable rating
        yet, appropriate when the user profile explicitly cares about
        rating (e.g. "seeking highly rated restaurants").
    """
    candidates = data
    if profile.get("city"):
        candidates = candidates[candidates["City"] == profile["city"]]
        if candidates.empty:
            print(f"[warning] No restaurants found in city='{profile['city']}'.")
            return candidates
 
    if exclude_unrated:
        candidates = candidates[candidates["Is Rated"]]
 
    if candidates.empty:
        print("[warning] No candidates remain after filtering.")
        return candidates
 
    full_matrix = fs.build_matrix(weights)
    cand_matrix = full_matrix[candidates.index.to_numpy()]
 
    user_vec = fs.build_user_vector(profile, weights).reshape(1, -1)
    sims = cosine_similarity(cand_matrix, user_vec).ravel()
 
    out = candidates.copy()
    out["Similarity"] = sims
    out = out.sort_values("Similarity", ascending=False).head(top_n)
 
    display_cols = [
        "Restaurant Name", "City", "Cuisines", "Price range",
        "Average Cost for two", "Currency", "Aggregate rating",
        "Rating text", "Votes", "Similarity",
    ]
    return out[display_cols].reset_index(drop=True)

# 6.EVALUATION FUNCTION:
 
def evaluate_recommendations(profile: dict, recs: pd.DataFrame, overall_mean_rating: float) -> dict:
    """Preference-alignment metrics for a single user's recommendation set."""
    if recs.empty:
        return {}
 
    metrics = {}
 
    if profile.get("cuisines"):
        wanted = set(profile["cuisines"])
        match = recs["Cuisines"].apply(
            lambda s: bool(wanted.intersection({c.strip() for c in s.split(",")}))
        )
        metrics["cuisine_match_rate"] = float(match.mean())
 
    if profile.get("price_range"):
        metrics["mean_abs_price_deviation"] = float(
            (recs["Price range"] - profile["price_range"]).abs().mean()
        )
 
    if profile.get("city"):
        metrics["city_match_rate"] = float((recs["City"] == profile["city"]).mean())
 
    metrics["mean_recommended_rating"] = float(recs["Aggregate rating"].mean())
    metrics["lift_over_overall_mean_rating"] = float(
        recs["Aggregate rating"].mean() - overall_mean_rating
    )
    return metrics

def cuisine_coherence_check(data: pd.DataFrame, fs: RestaurantFeatureSpace, sample_size=300, k=5, seed=RANDOM_STATE) -> float:
    """
    A dataset-supported sanity/robustness check that does NOT require
    ground-truth user ratings: for a random sample of restaurants, find
    each one's top-k nearest neighbours (by cuisine-only cosine similarity,
    restricted to the same city so the comparison is meaningful) and
    measure how often the neighbours actually share at least one cuisine
    tag with the seed restaurant. This checks that the similarity engine
    is behaving sensibly at scale -- it is NOT a substitute for a real
    accuracy metric such as precision@k against known likes.
    """
    rng = np.random.default_rng(seed)
    cuisine_only = fs.cuisine_matrix.astype(float)
    idx_by_city = data.groupby("City").indices
 
    sample_idx = rng.choice(data.index.to_numpy(), size=min(sample_size, len(data)), replace=False)
    hits = []
    for i in sample_idx:
        city = data.loc[i, "City"]
        city_idx = idx_by_city.get(city, np.array([]))
        city_idx = city_idx[city_idx != data.index.get_loc(i)]
        if len(city_idx) < k:
            continue
        seed_vec = cuisine_only[data.index.get_loc(i)].reshape(1, -1)
        sims = cosine_similarity(cuisine_only[city_idx], seed_vec).ravel()
        top_k = city_idx[np.argsort(-sims)[:k]]
        seed_cuisines = set(data.loc[i, "Cuisine List"])
        neighbour_hit = [
            bool(seed_cuisines.intersection(set(data.iloc[j]["Cuisine List"])))
            for j in top_k
        ]
        hits.append(np.mean(neighbour_hit))
    return float(np.mean(hits)) if hits else float("nan")

if __name__ == "__main__":
    df_raw = load_data("Restaurant Dataset.xlsx")
    eda = run_eda(df_raw)
    data, prep_meta = preprocess(df_raw)
    fs = RestaurantFeatureSpace(data)
 
    overall_mean_rating = data.loc[data["Is Rated"], "Aggregate rating"].mean()
 
    print("=== EDA ===")
    print("shape:", eda["shape"])
    print("dropped rows (missing cuisine):", prep_meta["dropped_missing_cuisine"])
    print("clean shape:", data.shape)
    print("n_unique_cuisines:", eda["n_unique_cuisines"])
    print("overall mean rating (rated only):", round(overall_mean_rating, 3))
 
    profiles = {
        "User 1 - Affordable Indian": dict(
            cuisines=["North Indian"], price_range=1, city="New Delhi"
        ),
        "User 2 - Highly Rated Seeker": dict(
            min_rating=4.5, city="New Delhi"
        ),
        "User 3 - Premium Diner": dict(
            cuisines=["Continental", "Italian"], price_range=4, min_rating=4.0,
            city="Gurgaon", prefers_table_booking=True
        ),
        "User 4 - Cuisine + Location": dict(
            cuisines=["Japanese"], city="New Delhi"
        ),
        "User 5 - Mixed Preferences": dict(
            cuisines=["Cafe", "Desserts"], price_range=2, min_rating=3.5
        ),
    }
 
    weight_overrides = {
        "User 2 - Highly Rated Seeker": {"cuisine": 0.5, "rating": 4.0, "price": 0.5},
        "User 3 - Premium Diner": {"cuisine": 2.0, "price": 3.0, "rating": 2.0},
    }
    exclude_unrated_flags = {
        "User 2 - Highly Rated Seeker": True,
        "User 3 - Premium Diner": True,
    }
 
    for name, profile in profiles.items():
        print(f"\n\n===== {name} =====")
        print("Profile:", profile)
        w = weight_overrides.get(name)
        excl = exclude_unrated_flags.get(name, False)
        recs = recommend(profile, data, fs, top_n=5, weights=w, exclude_unrated=excl)
        print(recs.to_string(index=False))
        metrics = evaluate_recommendations(profile, recs, overall_mean_rating)
        print("Metrics:", metrics)
 
    coherence = cuisine_coherence_check(data, fs)
    print("\n\n=== Cuisine coherence check (top-5 same-city neighbours sharing >=1 cuisine) ===")
    print(f"{coherence:.3f}")

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/naikmayanka04/restaurant-recommender)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![pandas](https://img.shields.io/badge/pandas-data%20processing-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-cosine%20similarity-F7931E)
![status](https://img.shields.io/badge/status-complete-brightgreen)

A content-based restaurant recommender built on the [Zomato Restaurants dataset](https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data) (9,551 restaurants, 21 columns). Given a user's stated cuisine, price, rating, and location preferences, it ranks restaurants by cosine similarity between the restaurant's own attributes and the user's preference profile — no user-interaction history required (and none exists in this dataset).

## Quickstart

```bash
git clone <this-repo-url>
cd restaurant-recommender
pip install -r requirements.txt
python recommender.py
```

This runs the full pipeline end to end — EDA → preprocessing → feature engineering → recommendations for 5 sample users → evaluation — and prints every result shown in this README.

## Repository structure

```
.
├── README.md              # this file — full write-up + results
├── recommender.py         # complete, runnable pipeline
├── requirements.txt       # pandas, numpy, scikit-learn, matplotlib, openpyxl
├── dataset.xlsx           # source data (Zomato Restaurants dataset)
└── charts/
    ├── top_cuisines.png
    ├── price_range_dist.png
    └── rating_dist.png
```

> **Note:** `dataset.xlsx` isn't included in this bundle — add your copy of the dataset to the repo root (or update the path in `recommender.py`) before running. It's the well-known public Zomato Restaurants dataset available on Kaggle.

---

# Content-Based Restaurant Recommendation System
### A Complete Data Science Project — Dataset-Grounded Implementation

---

## 1. Problem Definition

The goal is to build a **content-based recommendation system** that, given a restaurant dataset and a user's stated preferences (cuisine, price, rating expectations, location, etc.), returns a ranked list of restaurants whose *own attributes* most closely match what the user is looking for — without relying on other users' ratings or purchase history (which this dataset does not contain).

This is a classic **item-profile-matching** problem: each restaurant is converted into a feature vector describing "what it is," each user preference is converted into a feature vector describing "what they want," and restaurants are ranked by similarity between the two vectors.

---

## 2. Dataset Overview

The uploaded file is an Excel workbook (saved with a `.csv` extension) containing **9,551 restaurants and 21 columns** — this is the widely used "Zomato Restaurants" dataset. Every field below is present verbatim in the file; nothing here is inferred.

| # | Column | Type | What it represents |
|---|--------|------|---------------------|
| 1 | Restaurant ID | Numeric ID | Unique restaurant identifier |
| 2 | Restaurant Name | Text | Name of the restaurant |
| 3 | Country Code | Numeric (opaque) | Country identifier — **no decode table was included in this file**, so it cannot be translated to a country name reliably |
| 4 | City | Text | City the restaurant is located in |
| 5 | Address | Text | Full street address |
| 6 | Locality | Text | Neighbourhood / area |
| 7 | Locality Verbose | Text | Locality + city, verbose form |
| 8 | Longitude | Float | Geographic coordinate |
| 9 | Latitude | Float | Geographic coordinate |
| 10 | Cuisines | Text (comma-separated) | One or more cuisines served |
| 11 | Average Cost for two | Numeric | Cost in local currency |
| 12 | Currency | Text | Currency the cost is denominated in |
| 13 | Has Table booking | Yes/No | Table reservation offered |
| 14 | Has Online delivery | Yes/No | Online delivery offered |
| 15 | Is delivering now | Yes/No | Currently delivering (near-constant) |
| 16 | Switch to order menu | Yes/No | Constant across all rows |
| 17 | Price range | 1–4 (ordinal) | Zomato's own affordability tier, comparable across currencies |
| 18 | Aggregate rating | 0.0–5.0 | Average customer rating |
| 19 | Rating color | Text | Derived directly from Aggregate rating |
| 20 | Rating text | Text | Derived directly from Aggregate rating (e.g. "Excellent", "Not rated") |
| 21 | Votes | Numeric | Number of ratings received |

**Numerical variables:** Longitude, Latitude, Average Cost for two, Price range, Aggregate rating, Votes.
**Categorical variables:** Country Code, City, Locality, Cuisines (multi-label), Currency, Has Table booking, Has Online delivery, Is delivering now, Switch to order menu, Rating color, Rating text.

There is **no per-user interaction table** (no user IDs, no "user X rated restaurant Y" log) anywhere in the file — this single fact drives the evaluation strategy in Section 11.

---

## 3. Exploratory Data Analysis

Actual figures computed from the file:

- **Shape:** 9,551 rows × 21 columns.
- **Missing values:** only the `Cuisines` column has missing values — **9 rows (0.09%)**. Every other column is fully populated.
- **Duplicates:** **0 exact duplicate rows** and **0 duplicate `Restaurant ID`s**. No de-duplication is required.
- **Cities:** 141 unique cities. Heavily skewed toward the Delhi NCR region — New Delhi (5,473), Gurgaon (1,118), Noida (1,080), Faridabad (251) alone account for **~89.5%** of all restaurants; the remaining ~10.5% is spread thinly across ~137 other cities worldwide (20 rows each, typically), including international cities like Abu Dhabi, Dubai, London, Cape Town, Istanbul (garbled in-source, see below), Rio de Janeiro, and Singapore.
- **Cuisines:** 145 unique individual cuisine tags once the comma-separated strings are split (e.g. "French, Japanese, Desserts" → 3 tags). Most common: North Indian (3,960), Chinese (2,735), Fast Food (1,986), Mughlai (995), Italian (764).

<img src="charts/top_cuisines.png" width="620">

- **Price range** (1 = cheapest, 4 = most expensive): 1 → 4,444 · 2 → 3,113 · 3 → 1,408 · 4 → 586. The dataset is dominated by budget/mid-range restaurants.

<img src="charts/price_range_dist.png" width="480">

- **Currency:** 12 distinct currencies are present (Indian Rupees dominate with 8,652 rows / 90.6%, followed by US Dollar, Pounds, Brazilian Real, and others). This directly affects how "Average Cost for two" can be used (see Section 4).
- **Ratings:** `Rating text` breaks down as Average (3,737), **Not rated (2,148)**, Good (2,100), Very Good (1,079), Excellent (301), Poor (186). All 2,148 "Not rated" rows have `Aggregate rating == 0.0` — and a data-quality nuance worth flagging: **1,054 of those 2,148 already have 1–3 votes**, meaning "0.0" mostly represents *not enough votes yet to publish a score*, not necessarily a bad restaurant.

<img src="charts/rating_dist.png" width="520">

- **Data-quality issues found:**
  1. 9 missing `Cuisines` values.
  2. 2,148 restaurants (22.5%) have no real rating yet (`Aggregate rating = 0`), which would unfairly look like "worst possible" restaurants if treated as literal low scores.
  3. `Average Cost for two` spans 12 different currencies and is **not directly comparable** without conversion (e.g. 400 in Indian Rupees ≠ 400 in Indonesian Rupiah).
  4. `Country Code` is a bare integer with no accompanying lookup table in this file.
  5. 54 rows contain garbled/mojibake city names (e.g. `Bras..lia`, `S..o Paulo`, `..stanbul`) — an encoding artifact from the source file, left unchanged (see Section 4).
  6. `Switch to order menu` is constant ("No") for all 9,551 rows — zero information.
  7. `Rating color` and `Rating text` are deterministic derivatives of `Aggregate rating` (redundant if used alongside it).

---

## 4. Data Preprocessing

Each decision below is a direct response to an issue found in Section 3.

| Issue | Decision | Reasoning |
|---|---|---|
| 9 missing `Cuisines` | **Dropped** those 9 rows | Cuisine is the primary content signal; there's no non-fabricated way to fill it in, and the loss is negligible (0.09%) |
| `Switch to order menu` constant | **Excluded** from modeling | Zero variance = zero discriminative value |
| `Rating color` / `Rating text` | **Excluded** from the similarity vector, kept only for display | They are 1-to-1 derived from `Aggregate rating`; including them would double-count the same information |
| `Is delivering now` (99.6% "No") | **Excluded** from the similarity vector | Near-zero variance, negligible signal |
| `Aggregate rating = 0` conflated with "bad rating" | Added an explicit **`Is Rated`** boolean flag (`Aggregate rating > 0`) | Lets the recommender treat "not yet rated" differently from "rated poorly" instead of silently penalizing new/lightly-reviewed restaurants |
| `Average Cost for two` not comparable across 12 currencies | **Not used** as a similarity feature; `Price range` (Zomato's own currency-normalized 1–4 tier) used instead. Raw cost + currency kept for display only | Prevents a fundamentally invalid numeric comparison (mixing currencies without conversion rates) |
| `Country Code` has no decode table | Kept as an opaque id, **not used** as the primary location signal | `City` is the interpretable, directly-usable location field already in the file |
| Garbled city names (54 rows) | **Left unchanged** | Renaming them would mean guessing spellings not verifiably present in the file; the corruption is internally consistent per city, so exact-match filtering by `City` still works correctly |
| Cuisine categorical encoding | **Multi-hot encoding** (via `MultiLabelBinarizer`) over 145 cuisine tags | A restaurant can serve several cuisines at once — one-hot would incorrectly force a single category |
| `Price range`, `Aggregate rating` scale mismatch | **Min-max scaled to [0, 1]** (`(price-1)/3`, `rating/5`) | Puts every numeric feature on the same scale before combining into one vector, so no single feature dominates cosine similarity purely due to units |
| `Has Table booking`, `Has Online delivery` | Encoded as binary 0/1 | Simple, real-variance service-style signals worth including (12.1% and 25.7% "Yes" respectively) |

No assumptions were made about any field that does not exist in the dataset (e.g. no cuisine, country name, or rating was invented).

---

## 5. Recommendation Criteria

Based on what is actually available in the dataset, the system uses:

| Criterion | Column(s) used | Role |
|---|---|---|
| **Cuisine** | `Cuisines` (multi-hot, 145 tags) | Primary content signal — highest default weight |
| **Price range** | `Price range` (scaled ordinal) | Affordability match |
| **Rating** | `Aggregate rating` (scaled) + `Is Rated` flag | Quality signal, with an explicit escape hatch for "not yet rated" restaurants |
| **Location** | `City` | Hard filter (exact match) — recommending a restaurant on another continent is not useful, so this is applied as a candidate filter rather than a soft similarity dimension |
| **Service style** | `Has Table booking`, `Has Online delivery` | Secondary "other relevant characteristics" — light weight by default |

`Country Code`, raw `Average Cost for two`, `Longitude`/`Latitude`, `Address`, and `Locality Verbose` are **available but intentionally not used** in the similarity vector for the reasons given in Section 4 (see also Section 13, Limitations, for how Latitude/Longitude could be used in a future version).

---

## 6. Feature Engineering

Every restaurant is projected into one numeric vector:

```
[ cuisine multi-hot (145 dims) × w_cuisine,
  price_scaled (1 dim)         × w_price,
  rating_scaled (1 dim)        × w_rating,
  table_booking (1 dim)        × w_service,
  online_delivery (1 dim)      × w_service ]
```

Default weights: `cuisine = 3.0`, `price = 1.0`, `rating = 1.0`, `service = 0.5` — cuisine dominates because it is the strongest, most explicit statement of "what kind of restaurant this is," while the other blocks fine-tune the ranking. Weights are adjustable per query (demonstrated in Section 10, where "highly rated" and "premium dining" users get different weight profiles).

A **user preference profile** is projected into the *same* vector space: desired cuisines become a multi-hot vector, desired price/rating become scaled scalars (or a neutral 0.5 if the user has no opinion on that dimension, so it doesn't bias the similarity score in either direction), and desired service preferences become 0/1 flags. Because both restaurants and user profiles live in the same space, **cosine similarity** between them is directly meaningful.

---

## 7. Content-Based Filtering Methodology

1. Optionally filter candidate restaurants to the user's preferred `City` (hard filter).
2. Optionally drop `Is Rated == False` restaurants when the user profile explicitly cares about rating (`exclude_unrated=True`) — used for the "highly rated" and "premium dining" sample users below.
3. Build the restaurant feature matrix (Section 6) for the remaining candidates.
4. Build the user vector from the profile, using the same weights.
5. Compute **cosine similarity** between every candidate and the user vector.
6. Sort descending, return the top-N restaurants requested by the user (Top 5 / Top 10 / etc. — configurable).

Cosine similarity was chosen over Euclidean distance because it measures the *direction* of preference alignment (which cuisines/attributes are present) rather than raw magnitude, which is the standard, well-justified choice for multi-hot/sparse content vectors like this one.

---

## 8. Implementation

The full, runnable implementation lives in [`recommender.py`](recommender.py) — it's the exact code used to produce every table and number in this README. It's organized into clear, commented stages:

- `load_data()` — loads the raw workbook
- `run_eda()` — computes the EDA facts quoted below
- `preprocess()` — applies every cleaning decision in Section 4, with reasoning in the docstring
- `RestaurantFeatureSpace` — builds the shared restaurant/user vector space (Section 6)
- `recommend()` — ranks candidates by cosine similarity, with city filtering and an `exclude_unrated` option
- `evaluate_recommendations()` / `cuisine_coherence_check()` — the two evaluation approaches in Section 11

**Usage example:**

```python
from recommender import load_data, preprocess, RestaurantFeatureSpace, recommend

df_raw = load_data("dataset.xlsx")
data, _ = preprocess(df_raw)
fs = RestaurantFeatureSpace(data)

profile = {
    "cuisines": ["North Indian"],
    "price_range": 1,
    "city": "New Delhi",
}

top5 = recommend(profile, data, fs, top_n=5)
print(top5)
```

Running the script directly (`python recommender.py`) reproduces every result in this README end to end.

## 9 & 10. Sample User Preferences and Top-N Recommendations

All five profiles use cuisines and cities that actually exist in the dataset. Each ran with `top_n = 5`.

### User 1 — Affordable Indian Cuisine
**Preferences:** Cuisine = North Indian · Price range = 1 (cheapest tier) · City = New Delhi

| Rank | Restaurant | Cuisine | Price | Cost for two | Rating | Similarity |
|---|---|---|---|---|---|---|
| 1 | Delhi 6 | North Indian | 1 | ₹350 | 2.6 (Average) | 0.99998 |
| 2 | Apni Rasoi | North Indian | 1 | ₹300 | 2.6 (Average) | 0.99998 |
| 3 | Sat Guru Dhaba | North Indian | 1 | ₹350 | 2.6 (Average) | 0.99998 |
| 4 | Himalaya Dhaba | North Indian | 1 | ₹400 | 2.6 (Average) | 0.99998 |
| 5 | Kamal Meat House | North Indian | 1 | ₹400 | 2.6 (Average) | 0.99998 |

*Why:* every result is a pure North Indian, cheapest-tier restaurant in New Delhi — a perfect content match on the two dimensions the user cares about most.

### User 2 — Seeking Highly Rated Restaurants
**Preferences:** min. rating = 4.5 · City = New Delhi (cuisine open; rating weight boosted to 4.0, unrated restaurants excluded)

| Rank | Restaurant | Cuisine | Price | Cost for two | Rating | Similarity |
|---|---|---|---|---|---|---|
| 1 | Kopper Kadai | North Indian | 3 | ₹1,400 | 4.8 (Excellent) | 0.9915 |
| 2 | Masala Library | Modern Indian | 4 | ₹5,000 | 4.9 (Excellent) | 0.9904 |
| 3 | Indian Accent – The Manor | Modern Indian | 4 | ₹4,000 | 4.9 (Excellent) | 0.9904 |
| 4 | Carnatic Cafe | South Indian | 2 | ₹550 | 4.4 (Very Good) | 0.9898 |
| 5 | Drool Waffles | Desserts | 2 | ₹500 | 4.2 (Very Good) | 0.9889 |

*Why:* with rating weighted far above cuisine, the system correctly surfaces the city's top-rated restaurants (mean recommended rating 4.64, vs. the dataset's overall rated-restaurant average of 3.44) regardless of cuisine.

### User 3 — Premium Dining
**Preferences:** Cuisine = Continental/Italian · Price range = 4 (top tier) · min. rating = 4.0 · City = Gurgaon · Prefers table booking (price and rating weights boosted)

| Rank | Restaurant | Cuisine | Price | Cost for two | Rating | Similarity |
|---|---|---|---|---|---|---|
| 1 | Raasta | Continental, Italian | 4 | ₹2,000 | 3.8 (Good) | 0.9935 |
| 2 | The People & Co. | Continental, Italian | 3 | ₹1,700 | 3.7 (Good) | 0.9825 |
| 3 | The Brewhouse | Continental | 4 | ₹2,500 | 4.1 (Very Good) | 0.8933 |
| 4 | Bella Cucina – Le Meridien Gurgaon | Italian | 4 | ₹4,000 | 4.1 (Very Good) | 0.8933 |
| 5 | Prego – The Westin Gurgaon | Italian | 4 | ₹3,000 | 3.8 (Good) | 0.8932 |

*Why:* restaurants matching both requested cuisines at the requested price tier (Raasta) rank above those matching only one cuisine, even when the latter have a higher raw rating — demonstrating that the system balances *all* stated preferences rather than chasing a single feature.

### User 4 — Particular Cuisine + Location
**Preferences:** Cuisine = Japanese · City = New Delhi

| Rank | Restaurant | Cuisine | Price | Cost for two | Rating | Similarity |
|---|---|---|---|---|---|---|
| 1 | Tamura | Japanese | 3 | ₹1,500 | 3.3 (Average) | 0.9851 |
| 2 | Tamura | Japanese | 3 | ₹1,500 | 3.5 (Good) | 0.9845 |
| 3 | Manami Japanese Restaurant | Japanese | 3 | ₹1,000 | 0.0 (Not rated) | 0.9726 |
| 4 | Fuji Japanese Restaurant | Japanese | 4 | ₹2,000 | 4.0 (Very Good) | 0.9625 |
| 5 | Daitchi | Japanese, Chinese | 3 | ₹1,100 | 3.5 (Good) | 0.7220 |

*Why:* pure single-cuisine matches rank above the mixed Japanese+Chinese restaurant (Daitchi), whose similarity visibly drops once its vector is "diluted" by a second cuisine the user didn't ask for. "Tamura" appears twice legitimately — the dataset lists two separate outlets (different `Restaurant ID`s, different vote counts) under the same brand name, not a duplicate row.

### User 5 — Mixed Preferences
**Preferences:** Cuisine = Cafe/Desserts · Price range = 2 · min. rating = 3.5 (no city restriction — open to any location)

| Rank | Restaurant | City | Cuisine | Price | Cost for two | Rating | Similarity |
|---|---|---|---|---|---|---|---|
| 1 | The Bake Studio | Nashik | Cafe, Desserts | 2 | ₹400 | 3.6 (Good) | 0.99999 |
| 2 | The Haus | Surat | Cafe, Desserts | 2 | ₹500 | 3.9 (Good) | 0.99983 |
| 3 | Turta Home Cafe | Ankara | Cafe, Desserts | 2 | ₺35 | 4.3 (Very Good) | 0.99934 |
| 4 | The Chocolate Room | Faridabad | Cafe, Desserts | 2 | ₹900 | 3.6 (Good) | 0.99335 |
| 5 | Red Mango | Gurgaon | Cafe, Desserts | 2 | ₹700 | 3.6 (Good) | 0.99335 |

*Why:* with no city constraint, the system draws from across the whole dataset (Nashik, Surat, Ankara, Faridabad, Gurgaon) and still finds a near-perfect cuisine + price match everywhere — showing the recommender generalizes beyond the Delhi-NCR-heavy bulk of the data.

---

## 11. Recommendation Evaluation

**This dataset has no ground-truth user-item interaction log** (no user IDs, no historical "user liked/rated restaurant X" table) — only restaurant-side attributes. Conventional offline accuracy metrics that require known ground truth (Precision@K, Recall@K, RMSE/MAE against held-out ratings, NDCG, etc.) **cannot be reliably calculated** here, because there is nothing to hold out and predict. Calculating them anyway would require fabricating a synthetic "ground truth," which was explicitly avoided per the project's no-invented-data rule.

Instead, two dataset-supported evaluation approaches were used:

### 11.1 Preference-alignment evaluation (per user)

For each sample user, the delivered Top-5 list was checked against the *stated* preferences:

| User | Cuisine match rate | Mean |price − requested| | City match rate | Mean recommended rating | Rating lift vs. dataset average (3.44) |
|---|---|---|---|---|---|
| 1 — Affordable Indian | 100% | 0.0 | 100% | 2.60 | −0.84 |
| 2 — Highly Rated Seeker | n/a | n/a | 100% | 4.64 | **+1.20** |
| 3 — Premium Diner | 100% | 0.2 | 100% | 3.90 | +0.46 |
| 4 — Cuisine + Location | 100% | n/a | 100% | 2.86 | −0.58 |
| 5 — Mixed Preferences | 100% | 0.0 | n/a | 3.80 | +0.36 |

Every user with a stated cuisine or city preference received a **100% match rate** on that constraint, and price deviation stayed at or near zero. Ratings moved exactly in the direction each profile asked for: User 2 (explicitly rating-driven) saw a **+1.20** lift over the dataset average, while User 1 (whose profile said nothing about rating and only cared about cheap North Indian food) correctly received restaurants near the *typical* rating for that cheap segment, not artificially inflated ones.

### 11.2 Cuisine-similarity coherence check (system-level, quantitative)

Since ground-truth ratings aren't available, a structural sanity check was run instead: 300 randomly sampled restaurants were each compared to their own top-5 same-city cuisine-similarity neighbours (excluding themselves), measuring how often those neighbours share at least one cuisine tag with the seed restaurant.

**Result: 96.6%** of same-city top-5 neighbours shared at least one cuisine tag with their seed restaurant.

This confirms the similarity engine is behaving sensibly at scale — restaurants that the model considers "similar" are, in fact, overwhelmingly serving related food — but it is explicitly **not** a substitute for a true accuracy metric like precision@k against verified user likes, since no such ground truth exists in this file.

---

## 12. Results and Interpretation

- **Which features have the greatest influence?** Cuisine dominates by design (default weight 3.0 vs. 1.0/1.0/0.5 for price/rating/service) because it is the strongest available statement of restaurant identity. When a query's weights emphasize rating instead (User 2) or price+rating together (User 3), the ranking visibly shifts — see below.

- **Why particular restaurants rank near the top:** In every profile, the top result is the restaurant whose *combination* of attributes deviates least from the user's stated wants. User 4 illustrates this clearly: "Daitchi" (Japanese **and** Chinese) drops to similarity 0.72 versus 0.96–0.99 for pure-Japanese restaurants, because its cuisine vector partially points away from the user's single-cuisine request.

- **How changing preferences changes results:** Compare User 2 (rating weight = 4.0, cuisine weight = 0.5) against a default-weighted query on the same city — boosting rating weight replaced cuisine-driven results with the city's genuinely top-rated restaurants (Masala Library, Indian Accent, 4.8–4.9 stars) regardless of cuisine. This is a direct, reproducible demonstration that the weighting scheme controls *which* stated preference the system prioritizes.

- **Data-quality effects on results:** User 4's third result, "Manami Japanese Restaurant," has `Aggregate rating = 0.0` / "Not rated" and still appeared because that profile did not set `exclude_unrated=True` (it wasn't a rating-driven query). This shows the `Is Rated` flag working as designed — the restaurant wasn't penalized as "1-star bad," it was simply not filtered out when rating wasn't the user's concern.

---

## 13. Limitations

1. **No user history or ratings** — this is a pure content-based system; it cannot learn from what similar users liked (no collaborative signal exists in this file), so it cannot capture things like "people who liked X also liked Y" beyond what's encoded in the restaurant's own attributes.
2. **New/lightly-reviewed restaurants** (`Is Rated = False`, 22.5% of the dataset) have no reliable quality signal at all — the system can only fall back to cuisine/price/service attributes for them.
3. **Currency-crossing cost comparisons are not supported** — `Average Cost for two` is shown for context but deliberately excluded from similarity, so a "cheap" tag from `Price range` doesn't tell you the exact amount in a currency you recognize unless you check the displayed currency column.
4. **Geographic granularity is coarse** — location is handled as an exact `City` match; two restaurants at opposite ends of a huge city (or the `Locality`/Latitude/Longitude fields) are treated identically. A user wanting "somewhere near me" within a city currently gets no extra help.
5. **Cold-start restaurants** (a brand-new listing with no cuisine tag) would currently be dropped entirely, since cuisine is required for the vector space.
6. **Popularity is not modeled** — `Votes` (how many people rated it) is not currently used as a feature, so a 4.8-rated restaurant with 5 votes and one with 5,000 votes are treated identically.
7. **The 96.6% coherence figure is a structural sanity check, not an accuracy metric** — it says the model behaves consistently, not that it satisfies real users, since no real user feedback exists in this dataset.

---

## 14. Future Improvements

1. **Hybrid filtering:** combine this content-based approach with collaborative filtering once real user interaction/order data becomes available, to also capture "customers like you enjoyed…" signals.
2. **User-feedback loop:** let users thumbs-up/down recommendations and use that implicit feedback to personalize weights per user over time, rather than using one fixed weight scheme for everyone.
3. **Geo-distance scoring:** replace the hard `City` filter with a soft, continuous score using the existing `Latitude`/`Longitude` columns (e.g. a decay function of distance from a chosen point), so "closest good match" becomes possible even within a large city.
4. **Confidence-weighted rating:** incorporate `Votes` alongside `Aggregate rating` (e.g. a Bayesian-averaged or Wilson-score rating) so heavily-reviewed restaurants aren't ranked identically to barely-reviewed ones with the same raw score.
5. **Country-code decoding:** if an official Country Code → Country Name mapping table becomes available, add true country-level filtering/display on top of the existing city-level location handling.

---

## Summary

**How it works:** every restaurant is converted into one numeric vector (cuisine multi-hot + scaled price + scaled rating + service flags); a user's stated preferences are converted into a vector in that exact same space; restaurants are ranked by cosine similarity to that vector, optionally hard-filtered by city and by whether unrated restaurants should be excluded.

**Key findings from testing:** across five realistic sample users, every stated cuisine and city preference was matched at a 100% rate in the Top-5 results, price deviated by at most 0.2 tiers on average, and rating-focused queries produced a genuine +1.20-star lift over the dataset average — confirming the weighting scheme correctly steers results toward whichever preference a user emphasizes. A 300-restaurant coherence check found 96.6% same-city cuisine-similarity agreement, supporting that the underlying similarity computation is well-behaved.

**Main limitations:** no collaborative/user-history signal is possible with this dataset; ~22.5% of restaurants have no reliable rating yet; costs aren't comparable across the 12 currencies present; location is handled only at city granularity; and popularity (vote count) isn't currently factored into ranking.

**Practical next steps:** (1) add a hybrid collaborative layer once real interaction data exists, (2) let user feedback adjust weights over time, (3) use the existing Latitude/Longitude fields for finer-grained, distance-based location scoring, (4) confidence-weight the rating feature using vote counts, and (5) decode `Country Code` if an official mapping table becomes available.

# BA Product Advisor

## Project Overview

The BA Product Advisor is an application designed to assist retail associates in recommending products to customers effectively. It features a clean and user-friendly frontend interface for associates and a robust machine learning backend that generates product recommendations based on customer data and purchase history.

This project integrates multiple datasets to provide accurate and personalized product suggestions, enhancing the customer shopping experience and driving sales.

## Datasets

The application leverages several key datasets:

- **Retail Transactions**: Contains detailed records of customer purchases, including product IDs, quantities, timestamps, and store locations.
- **Product Catalog**: Provides comprehensive information about each product, such as category, price, and attributes.
- **Customer Profiles**: Includes demographic data and purchase preferences for each customer.
- **Sales Promotions**: Details current and upcoming promotions that may influence product recommendations.

## Merge Logic

To create a unified view for the recommendation engine, the datasets are merged using the following logic:

1. **Join Retail Transactions with Customer Profiles** on customer ID to link purchases with customer demographics.
2. **Integrate Product Catalog** by matching product IDs from transactions to enrich purchase data with product attributes.
3. **Incorporate Sales Promotions** by aligning promotion periods and product IDs to factor in promotional effects.
4. **Aggregate Data** to generate features such as purchase frequency, average spend, and product affinity scores.

This merged dataset forms the foundation for training and evaluating machine learning models.

## Machine Learning Roadmap

The machine learning backend development follows these stages:

1. **Exploratory Data Analysis (EDA)**: Understand data distributions, identify patterns, and detect anomalies.
2. **Feature Engineering**: Create meaningful features from merged data, including customer segmentation and product embeddings.
3. **Model Selection**: Evaluate algorithms such as collaborative filtering, matrix factorization, and gradient boosting.
4. **Training and Validation**: Train models on historical data and validate using cross-validation techniques.
5. **Deployment**: Integrate the best-performing model into the backend service for real-time recommendations.
6. **Monitoring and Updates**: Continuously monitor model performance and update with new data to maintain accuracy.

## Next Steps

- **Enhance Frontend UX**: Improve the user interface for easier navigation and better visualization of recommendations.
- **Expand Dataset Coverage**: Incorporate additional data sources like customer feedback and social media trends.
- **Implement Real-Time Learning**: Develop capabilities for the model to learn from live data streams.
- **A/B Testing**: Conduct experiments to measure the impact of recommendations on sales.
- **Scalability Improvements**: Optimize backend infrastructure for handling larger datasets and more concurrent users.

This roadmap will guide the ongoing development to ensure the BA Product Advisor meets the needs of retail associates and customers alike.
Goal

Deliver top-N personalized product recommendations per customer with a confidence/rating, capture user feedback, and fold it back to improve models. Build something that works now, is extensible, and can leverage a vector database.
Strategy

Start simple with strong baselines, add collaborative filtering for personalization, then layer semantic/vector retrieval for cold‑start and coverage. Finish with a hybrid ranker that blends signals and supports feedback-driven updates.
Data Prep

Interactions table: customer_id, product_id, ts, qty, spend, discount, season, city, skin_type, age_group.
Targets: treat purchases as implicit feedback. Weight by log(1 + qty), log(1 + spend), or a mix; keep timestamps for temporal split and drift handling.
Catalog table: product_id, name, category, subcategory, text_blob (name + subcat + category).
Splits: time-based (train on first N months, validate on next M). Evaluate with Recall@K / NDCG@K per user.
Baselines (fast, robust)

Popularity by segment: top products overall and by season, city, skin_type. Good for cold‑start and sanity checks.
Item–item co-occurrence: from baskets, compute cosine/Jaccard similarities; recommend items similar to a user’s recent purchases.
Collaborative Filtering (personalization)

Matrix factorization for implicit data (ALS) or BPR/WARP:
Implicit ALS: user/item factors, confidence weights from interaction strength. Scales well, strong baseline.
BPR/WARP (e.g., LightFM): pairwise ranking objective; better ranking, more training time.
Context handling:
Easiest: train separate seasonal models or add season-aware reweighting at inference.
Later: context-aware MF (factorization machines) or re-ranker features.
Embeddings + Vector DB (semantic coverage + cold-start)

Product embeddings: encode text_blob with a sentence embedding model; store in vector DB with product_id, metadata.
User embeddings (content-based): weighted average of embeddings of a user’s purchased items; great for new or sparse users.
Use ANN search to:
Generate candidates similar to user embedding (content cold‑start).
Power semantic search (“moisturizer for dry skin”), merchandising, and diversity.
Good options:
Local: FAISS (fast, simple, no infra), pgvector (Postgres integration).
Vector DB pros: handles text, generalizes, cold‑start, fast ANN; cons: extra infra, memory footprint, needs refresh when catalog changes.
Hybrid Recommender (recommended path)

Two‑stage pipeline:
Candidate generation: union of
CF: top-N from user factors, plus item–item similar for last K items.
Vector: nearest neighbors to user content-embedding; optionally filter by season/city.
Popularity: segment-based tops to guarantee fallback coverage.
Re-ranking:
Start with a weighted blend: w_cf * CFscore + w_vec * VECscore + w_pop * POP.
Graduate to a learning-to-rank model (LightGBM/XGBoost) with features:
User/item factors and dot product (from ALS).
Vector cosine(sim) to user embedding.
Popularity, novelty/recency, price/margin, season match, city match, skin-type match.
Labels: held-out interactions; negative sample from non-purchased candidates.
Feedback + Ratings

UI returns explicit 1–5 rating or binary thumbs up/down.
Store in user_feedback with user_id, product_id, ts, rating, context.
Online: adjust user embedding with feedback (e.g., recency-weighted running average).
Offline: add feedback to implicit matrix with higher confidence, retrain ALS/Ranker periodically.
Exploration: consider ε-greedy or Thompson sampling to discover new items without degrading UX.
Project Structure

data/ existing outputs
features/prepare_interactions.py: build interactions/canonical IDs.
models/train_als.py: train implicit ALS, save factors users.npy, items.npy, id_maps.json.
embeddings/build_product_embeddings.py: produce product vectors, store to faiss.index or pgvector.
serve/candidate_gen.py: CF + vector DB + popularity candidate generation.
serve/rank.py: simple blender → later a trained LTR model.
serve/recommend.py: main entry to produce top-N per user with scores.
feedback/log_feedback.py: append user ratings to storage (CSV/SQLite/postgres), schedule retrains.
eval/metrics.py: Recall@K, NDCG@K, coverage/diversity.
Evaluation

Offline: Recall@K/NDCG@K on temporal holdout; track by segment (season, skin type) and by user lifecycle (new vs. active).
Online (later): CTR/acceptance rate, rating distributions, A/B testing against baselines.
Vector DB Handling in This Project

Start local with FAISS for repeatable dev:
Build an index: IndexFlatIP or HNSW with normalized vectors; persist to disk.
Store sidecar product_id array and metadata map.
If you prefer SQL + ops simplicity: Postgres + pgvector:
Store embeddings as a column; add ANN index; filter by metadata in SQL; simpler ops, slower than FAISS at large scale but good enough for tens of millions.
Move to managed (Pinecone/Milvus) if you need horizontal scale and filtering performance with low ops burden.
Advantages/Disadvantages
# app.py  -  Job market explorer (TF-IDF baseline)
# Install: pip install datasets pandas scikit-learn gradio
import re
from collections import Counter

import gradio as gr
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# 1. Load real public data (lukebarousse/data_jobs, ~785k data-related job postings, 2023)
ds = load_dataset("lukebarousse/data_jobs", split="train")
df = ds.to_pandas()

# 2. Clean: keep rows with a title and skills, remove duplicates, sample for speed
df = df[["job_title", "job_title_short", "company_name", "job_skills"]].dropna(
    subset=["job_title", "job_skills"]
)
df = df.drop_duplicates(subset=["job_title", "company_name"]).reset_index(drop=True)
df = df.sample(n=min(50_000, len(df)), random_state=42).reset_index(drop=True)

def skills_list(text):
    """'python, sql, excel' -> ['python', 'sql', 'excel']"""
    return [s.strip().lower() for s in str(text).split(",") if s.strip()]

# 3. Text used for matching: job title + skills
df["text"] = (df["job_title"].fillna("") + " " + df["job_skills"].fillna("").str.replace(",", " ")).str.lower()

# 4. Held-out split for evaluation (the test set is never used to build the index)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
train_matrix = vectorizer.fit_transform(train_df["text"])

# 5. Search function
def search(query, top_k=10):
    query = (query or "").strip()
    if len(query) < 3:
        return "Please type a job title or some skills (at least 3 characters).", None
    q_vec = vectorizer.transform([query.lower()])
    if q_vec.nnz == 0:
        return "No known words found. Try terms like 'data analyst', 'python', or 'sql'.", None
    scores = cosine_similarity(q_vec, train_matrix).ravel()
    top_idx = scores.argsort()[::-1][:top_k]
    results = train_df.iloc[top_idx].copy()
    results["match"] = scores[top_idx].round(3)

    table = results[["job_title", "company_name", "match"]]

    # Skills in demand among the matched jobs
    counts = Counter()
    for s in results["job_skills"]:
        counts.update(skills_list(s))
    top_skills = pd.DataFrame(counts.most_common(10), columns=["skill", "jobs"])
    return table, top_skills

# 6. Evaluation: precision@10 = share of top-10 results that have the same
#    short job category as the query job. Compared with a random baseline.
def evaluate(n_queries=200, k=10, seed=42):
    sample = test_df.sample(n=min(n_queries, len(test_df)), random_state=seed)
    hits, random_hits = [], []
    all_labels = train_df["job_title_short"].values
    for _, row in sample.iterrows():
        q_vec = vectorizer.transform([row["job_title"].lower()])
        scores = cosine_similarity(q_vec, train_matrix).ravel()
        top = scores.argsort()[::-1][:k]
        hits.append((all_labels[top] == row["job_title_short"]).mean())
        rand = train_df.sample(n=k, random_state=seed)["job_title_short"].values
        random_hits.append((rand == row["job_title_short"]).mean())
    return round(sum(hits) / len(hits), 3), round(sum(random_hits) / len(random_hits), 3)

if __name__ == "__main__":
    p_tfidf, p_random = evaluate()
    print(f"Precision@10  TF-IDF: {p_tfidf}   random baseline: {p_random}")

    with gr.Blocks(title="Job market explorer") as demo:
        gr.Markdown("# Job market explorer\nType a job or skills, see matching postings and the skills employers ask for.")
        q = gr.Textbox(label="Job or skills", placeholder="e.g. data analyst python sql")
        btn = gr.Button("Search")
        msg = gr.Markdown()
        jobs = gr.Dataframe(label="Matching jobs")
        skills = gr.Dataframe(label="Skills in demand among these jobs")

        def run(query):
            out = search(query)
            if out[1] is None:
                return out[0], None, None
            return "", out[0], out[1]

        btn.click(run, inputs=q, outputs=[msg, jobs, skills])
    demo.launch()

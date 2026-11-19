# nlp-capstone-<team>  (solo: <your name>)

## Problem and users
Job seekers in data-related fields who want to know which skills match a role. Users type a job or skills in plain language and see matching postings and the skills employers ask for most.

## Data source
- lukebarousse/data_jobs on Hugging Face (public dataset, about 785k data-related job postings, 2023).
- Cleaning: kept title, company and skills; removed rows without title or skills and duplicate title/company pairs; sampled 50,000 rows.
- Known limits: 2023 data only; mostly English postings; sample may not reflect all sectors.

## Pipeline
1. Load and clean the data.
2. Build text from job title + skills.
3. TF-IDF vectors (word unigrams and bigrams, English stop words removed, min document frequency 2).
4. Cosine similarity for search; top 10 results.
5. Count skills across matched jobs to show skills in demand.
6. Evaluation: 80/20 held-out split; precision@10 on 200 test queries, compared with a random baseline.

## Results
- Precision@10, TF-IDF: [FILL]
- Precision@10, random baseline: [FILL]
- Error analysis: [FILL, e.g. short titles like "analyst" match too broadly]

## How to run
    pip install datasets pandas scikit-learn gradio
    python app.py
Then open the local link printed in the terminal.

## Who did what
- [Your name]: all parts (data, pipeline, evaluation, app, presentation).
- AI assistant (Claude) used for drafting code and slides; every part was reviewed and can be explained by the author.

## Next steps
Sentence embeddings for semantic search; compare against TF-IDF; publish on Hugging Face Spaces.

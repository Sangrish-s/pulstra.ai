# arvix.py
import feedparser
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np

def fetch_arxiv_paper(paper):
    try:
        paper_id = paper.id.split('/')[-1]  # Extract the ID
        return {'id': paper_id, 'title': paper.title, 'abstract': paper.summary}
    except Exception as e:
        print(f"Error fetching {paper.title}: {e}")
        return {'id': '', 'title': paper.title, 'abstract': ''}

def fetch_and_process_papers(query_str, max_results=100):
    query_str = query_str.replace(' ', '+')
    base_url = f"http://export.arxiv.org/api/query?search_query={query_str}&sortBy=relevance&max_results={max_results}"
    papers = feedparser.parse(base_url).entries
    with ThreadPoolExecutor() as executor:
        return list(executor.map(fetch_arxiv_paper, papers))

def convert_titles_to_embeddings(titles):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(titles, show_progress_bar=True)
    return embeddings

def main(query_str):
    max_results = 100
    papers_data = fetch_and_process_papers(query_str, max_results)
    df = pd.DataFrame(papers_data)
    titles = df['title'].tolist()
    embeddings = convert_titles_to_embeddings(titles)
    df['embeddings'] = embeddings.tolist()
    df.to_csv("arxiv_papers_with_embeddings.csv", index=False)
    print("Data with embeddings fetched and saved.")

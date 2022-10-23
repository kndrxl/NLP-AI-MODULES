
import json
import os
import time
import pickle 
import hdbscan
import pandas as pd
import umap.umap_ as umap
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def load_pickle(name):
    pkl_object = pickle.load(open(name, 'rb'))
    return pkl_object

def umap_dim_reductor(sentence_embeddings, n_neighbors):
    umap_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, 
        n_components=2,
        min_dist=0.0, 
        metric='cosine'
    ).fit_transform(sentence_embeddings)
    return umap_embeddings

def hdbscan_clusterer(dim_reduced_embeddings, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',                      
        cluster_selection_method='eom'
        ).fit(dim_reduced_embeddings)
    return clusterer

def visualize_raw_df(dataframe, name_title):
    fig = px.scatter(
        dataframe, x='x', y='y',
        hover_data=['text'], 
        title = name_title, 
    )
    fig.show()

def visualize_clustered_df(dataframe, name_title):
    fig = px.scatter (
        dataframe, 
        x='x', 
        y='y',
        color='labels', 
        symbol='labels', 
        hover_data=['text'], 
        title=name_title,
        labels={"labels": "Label", "labels": "Label"}
    )
    fig.update_coloraxes(showscale=False)
    return fig

def save_to_folder(fig, path):
    fig.write_image(path)

def get_number_of_clusters(clustered_df):
    _labels = {}
    for row in clustered_df:
        label = f"Label_{row}" if row != -1 else "Outliers"
        _labels[label] =  _labels[label] + 1 if label in _labels else 1
    _labels = dict(sorted(_labels.items(), key=lambda item: item[1], reverse=True))
    clusters = len(list(_labels.items()))
    return clusters, _labels

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (
        df.groupby(['Topic']).text.count().reset_index().rename(
            {"Topic": "Topic", "text": "Size"}, 
            axis='columns'
            ).sort_values("Size", ascending=False
        )
    )
    return topic_sizes

def generate_metadata(json_file, path):
    with open(path, "w") as outfile:
        outfile.write(json.dumps(json_file, indent=4))
    return



if __name__ == "__main__":
    print("\nStarting Process...\n")
    start = time.time()
    df = pd.read_csv("tagalog_datasets/tagalog_newspapers/preprocessed_news_articles_136k.csv")
    df.drop('Unnamed: 0', inplace=True, axis=1)
    raw_corpus = df.Text
    corpus = df.Processed
    sentence_embeddings = load_pickle("tagalog_datasets/tagalog_newspapers/sentence_embeddings_preprocessed_136k.pkl")
    for i in range(50, 110, 10):
        umap_embeddings = umap_dim_reductor(sentence_embeddings, n_neighbors=i)
        umap_df = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
        umap_df['text'] = corpus
        for j in range(100, 1050, 50):
            clusterer = hdbscan_clusterer(umap_embeddings, min_cluster_size=j)
            clustered_result = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
            clustered_result['text'] = corpus
            clustered_result['labels'] = clusterer.labels_
            without_outliers = clustered_result[clustered_result.labels != -1]
            cluster_numbers, metadata = get_number_of_clusters(clustered_result.labels)

            if not os.path.exists(f"images/{i}_neighbors"):
                os.mkdir(f"images/{i}_neighbors")

            orig_fig = visualize_clustered_df(clustered_result, f"Neigbors: {i} | Min Cluster Size: {j} | Generated Clusters: {cluster_numbers-1}")
            orig_path = f"images/{i}_neighbors/{j}_min_cluster_size_with_outliers.png"
            save_to_folder(orig_fig, orig_path)
            fig = visualize_clustered_df(without_outliers, f"Neigbors: {i} | Min Cluster Size: {j} | Generated Clusters: {cluster_numbers-1}")
            path = f"images/{i}_neighbors/{j}_min_cluster_size_without_outliers.png"
            save_to_folder(fig, path)
            metadata_path = f"images/{i}_neighbors/{j}_min_cluster_size_metadata.json"
            generate_metadata(metadata, metadata_path)

            docs_df = pd.DataFrame(clustered_result, columns=["text"])
            docs_df['Topic'] = clusterer.labels_
            docs_df['Doc_ID'] = range(len(docs_df))
            docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'text': ' '.join})
            tf_idf, count = c_tf_idf(docs_per_topic.text.values, m=len(clustered_result))
            top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=50)
            top_words_path = f"images/{i}_neighbors/{j}_min_cluster_size_top_words.json"
            generate_metadata(top_n_words, top_words_path)

            print(f"\nDone Process\nNo. of Neighbors : {i}\nMinimum Cluster Size: {j}")



    end = time.time()
    print('Execution time:', end-start, 'seconds')
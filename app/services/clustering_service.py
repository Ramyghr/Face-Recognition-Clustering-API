#cluster_service.py
import os
import cv2
from app.core.config import settings
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx




def save_clustered_faces(faces: list, original_image, cluster_id: int, index: int):
    base_path = settings.CLUSTER_OUTPUT_PATH
    cluster_dir = os.path.join(base_path, f"person_{cluster_id}")
    cropped_dir = os.path.join(cluster_dir, "cropped_faces")
    originals_dir = os.path.join(cluster_dir, "images")

    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(originals_dir, exist_ok=True)

    cv2.imwrite(os.path.join(cropped_dir, f"face_{index}.jpg"), faces[index])
    cv2.imwrite(os.path.join(originals_dir, f"image_{index}.jpg"), original_image)
def cluster_embeddings(embeddings, threshold=0.4):
    """Hierarchical clustering with cosine distance"""
    if len(embeddings) < 2:
        return [0] * len(embeddings)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        affinity='cosine',
        linkage='average'
    )
    return clustering.fit_predict(embeddings)

def merge_clusters(embeddings, labels, merge_threshold=0.3):
    """Merge small clusters using cosine similarity"""
    unique_labels = set(labels)
    centroids = {}
    cluster_sizes = {}
    
    # Calculate centroids
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_embs = np.array(embeddings)[indices]
        centroids[label] = np.mean(cluster_embs, axis=0)
        cluster_sizes[label] = len(indices)
    
    # Merge small clusters
    new_labels = labels.copy()
    for current_label in unique_labels:
        if cluster_sizes[current_label] >= 2: 
            continue
            
        current_centroid = centroids[current_label].reshape(1, -1)
        best_sim = -1
        best_target = None
        
        for target_label in unique_labels:
            if target_label == current_label: 
                continue
            if cluster_sizes[target_label] < 2: 
                continue
                
            target_centroid = centroids[target_label].reshape(1, -1)
            sim = cosine_similarity(current_centroid, target_centroid)[0][0]
            if sim > best_sim:
                best_sim = sim
                best_target = target_label
                
        if best_target and best_sim > (1 - merge_threshold):
            new_labels[new_labels == current_label] = best_target
            
    return new_labels
def graph_based_merge(clusters, sim_threshold=0.9):
    """Final graph-based merging using connected components"""
    centroids = [np.mean(embeddings, axis=0) for embeddings, _ in clusters]
    centroids = np.vstack(centroids)
    
    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms
    
    # Build similarity graph
    similarity_matrix = cosine_similarity(centroids)
    G = nx.Graph()
    
    for i in range(len(clusters)):
        G.add_node(i)
    
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if similarity_matrix[i, j] >= sim_threshold:
                G.add_edge(i, j)
    
    # Find connected components
    components = list(nx.connected_components(G))
    
    # Merge clusters
    merged_clusters = []
    for comp in components:
        combined_embs = []
        combined_files = []
        for idx in comp:
            combined_embs.extend(clusters[idx][0])
            combined_files.extend(clusters[idx][1])
        merged_clusters.append((combined_embs, combined_files))
    
    return merged_clusters

def force_merge(clusters, threshold=0.15):
    """Aggressive final merging of tiny clusters"""
    if len(clusters) < 2:
        return clusters

    centroids = [np.mean(embeddings, axis=0) for embeddings, _ in clusters]
    centroids = np.vstack(centroids)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clusterer.fit_predict(centroids)

    merged = defaultdict(lambda: ([], []))
    for idx, label in enumerate(labels):
        merged[label][0].extend(clusters[idx][0])
        merged[label][1].extend(clusters[idx][1])

    return list(merged.values())


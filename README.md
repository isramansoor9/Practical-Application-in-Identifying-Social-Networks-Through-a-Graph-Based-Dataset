# Practical Application in Identifying Social Networks Through a Graph-Based Dataset

## Table of Contents

1.  [Overview](#overview)
2.  [Dataset](#dataset)
3.  [Requirements](#requirements)
4.  [Installation](#installation)
5.  [Usage](#usage)
6.  [Methodology](#methodology)
7.  [Results](#results)
8.  [Key Points](#key-points)
9.  [Challenges](#challenges)
10. [Fixing Challenges](#fixing-challenges)
11. [Discussion](#discussion)
12. [Conclusion](#conclusion)
13. [References](#references)
14. [License](#license)

## Overview

This project demonstrates the application of spectral clustering (a graph-based clustering method) to identify communities in social networks using the Facebook Social Circles Dataset. The dataset consists of anonymized social network data from Facebook, where nodes represent users and edges represent friendships or interactions. We construct graphs, perform eigenvalue decomposition on the Laplacian matrix, and apply K-Means clustering to detect social circles.

Key objectives:

*   Analyze social network structures.
*   Evaluate clustering quality using metrics like Silhouette Score, Modularity, Intra-Cluster Distance, and Inter-Cluster Distance.
*   Highlight challenges in graph-based clustering and potential solutions.

This repository contains a Jupyter notebook (`practical_application_in_identifying_social_networks_through_a_graph_based_dataset.ipynb`) that implements the full workflow, from data loading to evaluation and visualization.

## Dataset

The dataset is stored in the `./dataset` folder and includes files for 10 specific node IDs: `[0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]`. Each node has the following files:

*   `.edges`: List of edges (friendships) connected to the node.
*   `.circles`: Pre-defined social circles (ground truth communities).
*   `.egofeat`: Ego features for the central node.
*   `.feat`: Node features.
*   `.featnames`: Feature names.

The dataset is loaded into a dictionary for processing. Note: The original dataset can be sourced from SNAP: Stanford Network Analysis Project.

## Requirements

*   Python 3.12+
*   Libraries:
    *   `networkx` for graph creation and analysis.
    *   `matplotlib` for visualization.
    *   `numpy` and `scipy` for numerical computations.
    *   `scikit-learn` for clustering (KMeans), similarity measures (cosine similarity), and evaluation metrics (silhouette score, normalized mutual info score).
    *   `pandas` for data handling and tabular output.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/social-networks-graph-clustering.git
    cd social-networks-graph-clustering
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Sample `requirements.txt`:

    ```text
    networkx
    matplotlib
    numpy
    scipy
    scikit-learn
    pandas
    ```

3.  Download or place the dataset in the `./dataset` folder.

## Usage

1.  **Open the Jupyter notebook:**

    ```text
    jupyter notebook practical_application_in_identifying_social_networks_through_a_graph_based_dataset.ipynb
    ```

2.  **Run the cells sequentially:**

    *   Load libraries and dataset.
    *   Construct graphs and display visualizations.
    *   Perform spectral clustering.
    *   Evaluate and display results.

Example output: Network graphs for each node (limited to first 50 nodes for visualization) and a styled Pandas DataFrame with evaluation metrics.

## Methodology

1.  **Imports and Setup**

    Import necessary libraries for graph analysis, clustering, and visualization.

2.  **Loading Datasets**

    Load edge, circle, feature, and feature name files for each node ID into a dictionary.

3.  **Display Network Graph**

    *   Create a NetworkX graph from edges.
    *   Visualize a subgraph (first 50 nodes).
    *   Compute and print metrics: number of nodes/edges, average clustering coefficient.

<table>
  <tr>
    <td><img width="400" height="400" alt="image1" src="https://github.com/user-attachments/assets/2ba2b378-f6c3-4dec-a5e2-702de87dca2c" /></td>
    <td><img width="400" height="400" alt="image2" src="https://github.com/user-attachments/assets/9e5eb661-00f4-43ac-85d8-b4df82001500" /></td>
  </tr>
</table>



4.  **Spectral Clustering Process**

    *   Step 1: Construct similarity graph (adjacency matrix) from edges.  
    *   Step 2: Compute unweighted/weighted adjacency matrices.  
    *   Step 3: Build Laplacian matrix and perform eigenvalue decomposition.  
    *   Step 4: Select top-k eigenvectors for reduced-dimensional representation.  

    <table>
      <tr>
        <td align="center"><img width="400" height="260" alt="image1" src="https://github.com/user-attachments/assets/591bc875-53d3-4e4d-980e-1c0a6a0a83da" /><br><b>Linear Decision Boundary</b></td>
        <td align="center"><img width="400" height="260" alt="image2" src="https://github.com/user-attachments/assets/33cefe56-73b9-4237-8167-6bbc84292238" /><br><b>Non-Linear Decision Boundary</b></td>
      </tr>
    </table>

    *   Step 5: Apply K-Means clustering on eigenvectors to form communities.  
    *   Step 6: Evaluate clusters using multiple metrics.  


5.  **Evaluation**

    *   Silhouette Score: Measures cluster cohesion and separation.
    *   Modularity: Assesses community structure quality.
    *   Intra/Inter-Cluster Distances: Evaluate compactness and separation.

## Visualizing Clusters  

<table>
      <tr>
        <td align="center"><img width="794" height="812" alt="image" src="https://github.com/user-attachments/assets/45369a1e-bd4e-4b02-b579-4328380e069b" /><br><b>Linear Decision Boundary</b></td>
        <td align="center"><img width="794" height="812" alt="image" src="https://github.com/user-attachments/assets/2df76422-8a5c-484a-b337-d19ebf6528f6" /><br><b>Non-Linear Decision Boundary</b></td>
      </tr>
    </table>
    

## Results

### Summary of Findings

*   Node 3980: Highest Silhouette Score (0.99), indicating near-perfect clusters.
*   Node 686 & 698: High Silhouette Scores (0.96), but varying modularity.
*   Node 3437: Strong balance of Silhouette (0.91) and Modularity (0.60).
*   Lower modularity in some nodes (e.g., 686) suggests weaker community structures despite good cohesion.

### Evaluation Table

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/880b0abd-19c4-48eb-8c15-c167704c4726" />

(Values formatted to 2 decimal places; higher Silhouette/Modularity indicate better clustering.)

## Key Points

*   Model: Spectral clustering for community detection.
*   Graph Construction: Nodes as users, edges as interactions.
*   Dimensionality Reduction: Laplacian matrix and eigenvectors.
*   Clustering: K-Means on reduced space.
*   Applications: Marketing, sociology, network analysis.

## Challenges

*   Scalability: Eigenvalue decomposition is computationally expensive for large graphs.
*   Number of Clusters: Determining optimal k requires trial-and-error or indices.
*   Similarity Measure: Results depend on how similarities are defined.
*   Noise Sensitivity: Outliers can degrade clustering quality.
*   Graph Construction: Edge definitions greatly influence outcomes.

## Fixing Challenges

*   Scalability: Use approximate eigenvalue methods or efficient libraries.
*   Number of Clusters: Employ cluster validity indices (e.g., Silhouette) and domain knowledge.
*   Similarity Measure: Experiment with dataset-specific measures (e.g., cosine similarity).
*   Noise Sensitivity: Apply outlier detection and noise reduction preprocessing.
*   Graph Construction: Use robust methods like unweighted/weighted adjacency matrices.

## Discussion

Spectral clustering excels at detecting non-convex communities by capturing global graph patterns. However, it is computationally intensive and sensitive to noise/similarity definitions. Despite these, it is a powerful tool for social network analysis when optimized.

## Conclusion

This project showcases spectral clustering's effectiveness in uncovering social communities from graph data. Future enhancements could include advanced algorithms (e.g., Louvain method) for better scalability and accuracy.

## References

*   SNAP Dataset: Facebook Social Circles
*   NetworkX Documentation: networkx.org
*   Scikit-learn: Clustering Metrics

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

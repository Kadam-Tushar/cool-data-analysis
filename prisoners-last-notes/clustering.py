import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class GeminiClusteringAnalyzer:
    """
    Comprehensive clustering analysis for Gemini embeddings with visualization.
    """
    
    def __init__(self, embeddings_file="./data/gemini_embeddings_clustering.csv"):
        """
        Initialize the clustering analyzer.
        
        Args:
            embeddings_file (str): Path to the clustering embeddings CSV file
        """
        self.embeddings_file = embeddings_file
        self.df = None
        self.embeddings = None
        self.scaled_embeddings = None
        self.cluster_labels = None
        self.tsne_embeddings = None
        self.pca_embeddings = None
        
    def load_data(self):
        """Load and prepare the clustering embeddings data."""
        try:
            self.df = pd.read_csv(self.embeddings_file)
            print(f"Loaded {len(self.df)} rows from {self.embeddings_file}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Extract embedding columns (exclude metadata columns)
            embedding_cols = [col for col in self.df.columns if col.startswith('dim_')]
            self.embeddings = self.df[embedding_cols].values
            
            print(f"Embedding shape: {self.embeddings.shape}")
            print(f"Sample metadata columns: {[col for col in self.df.columns if not col.startswith('dim_')]}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_embeddings(self):
        """Preprocess embeddings for clustering."""
        # Remove any rows with NaN values
        valid_mask = ~np.isnan(self.embeddings).any(axis=1)
        self.embeddings = self.embeddings[valid_mask]
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"After removing NaN values: {len(self.embeddings)} rows")
        
        # Standardize embeddings
        scaler = StandardScaler()
        self.scaled_embeddings = scaler.fit_transform(self.embeddings)
        
        print("Embeddings standardized successfully")
        return True
    
    def find_optimal_clusters(self, max_clusters=20):
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            max_clusters (int): Maximum number of clusters to test
        """
        print("Finding optimal number of clusters...")
        
        # Test different numbers of clusters
        n_clusters_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for n_clusters in n_clusters_range:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_embeddings)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_embeddings, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(self.scaled_embeddings, cluster_labels))
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Elbow curve
        ax1.plot(n_clusters_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(n_clusters_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        # Calinski-Harabasz scores
        ax3.plot(n_clusters_range, calinski_scores, 'go-')
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Analysis')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal number of clusters
        optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
        
        return optimal_clusters
    
    def perform_clustering(self, n_clusters=8, method='kmeans'):
        """
        Perform clustering using specified method.
        
        Args:
            n_clusters (int): Number of clusters
            method (str): Clustering method ('kmeans', 'dbscan', 'agglomerative')
        """
        print(f"Performing {method} clustering with {n_clusters} clusters...")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            # For DBSCAN, we need to estimate eps
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors_fit = neighbors.fit(self.scaled_embeddings)
            distances, indices = neighbors_fit.kneighbors(self.scaled_embeddings)
            distances = np.sort(distances[:, 4])
            eps = np.percentile(distances, 90)
            clusterer = DBSCAN(eps=eps, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        self.cluster_labels = clusterer.fit_predict(self.scaled_embeddings)
        
        # Add cluster labels to dataframe
        self.df['cluster'] = self.cluster_labels
        
        # Calculate clustering metrics
        if method != 'dbscan' or len(set(self.cluster_labels)) > 1:
            silhouette = silhouette_score(self.scaled_embeddings, self.cluster_labels)
            calinski = calinski_harabasz_score(self.scaled_embeddings, self.cluster_labels)
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Calinski-Harabasz Score: {calinski:.4f}")
        
        # Show cluster distribution
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        print(f"\nCluster distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} statements")
        
        return True
    
    def perform_tsne(self, perplexity=30, n_iter=1000):
        """
        Perform t-SNE dimensionality reduction for visualization.
        
        Args:
            perplexity (int): t-SNE perplexity parameter
            n_iter (int): Number of iterations
        """
        print("Performing t-SNE dimensionality reduction...")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        self.tsne_embeddings = tsne.fit_transform(self.scaled_embeddings)
        
        print("t-SNE completed successfully")
        return True
    
    def perform_pca(self, n_components=2):
        """
        Perform PCA dimensionality reduction for visualization.
        
        Args:
            n_components (int): Number of PCA components
        """
        print("Performing PCA dimensionality reduction...")
        
        pca = PCA(n_components=n_components, random_state=42)
        self.pca_embeddings = pca.fit_transform(self.scaled_embeddings)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")
        
        return True
    
    def plot_clusters_tsne(self, figsize=(15, 10)):
        """Plot clusters using t-SNE visualization."""
        if self.tsne_embeddings is None:
            print("Please run perform_tsne() first")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Color-coded clusters
        scatter1 = ax1.scatter(self.tsne_embeddings[:, 0], self.tsne_embeddings[:, 1], 
                              c=self.cluster_labels, cmap='tab10', alpha=0.7, s=50)
        ax1.set_title('Clusters in t-SNE Space')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters", loc="upper right")
        ax1.add_artist(legend1)
        
        # Cluster sizes
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_counts)))
        
        bars = ax2.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
        ax2.set_title('Cluster Sizes')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Statements')
        ax2.set_xticks(range(len(cluster_counts)))
        ax2.set_xticklabels([f'Cluster {i}' for i in cluster_counts.index])
        
        # Add value labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_clusters_pca(self, figsize=(15, 10)):
        """Plot clusters using PCA visualization."""
        if self.pca_embeddings is None:
            print("Please run perform_pca() first")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Color-coded clusters
        scatter1 = ax1.scatter(self.pca_embeddings[:, 0], self.pca_embeddings[:, 1], 
                              c=self.cluster_labels, cmap='tab10', alpha=0.7, s=50)
        ax1.set_title('Clusters in PCA Space')
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters", loc="upper right")
        ax1.add_artist(legend1)
        
        # Explained variance
        pca = PCA(random_state=42)
        pca.fit(self.scaled_embeddings)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        ax2.set_title('Cumulative Explained Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_tsne(self):
        """Create interactive t-SNE plot with Plotly."""
        if self.tsne_embeddings is None:
            print("Please run perform_tsne() first")
            return
        
        # Create interactive scatter plot
        fig = px.scatter(
            x=self.tsne_embeddings[:, 0],
            y=self.tsne_embeddings[:, 1],
            color=self.cluster_labels,
            hover_data={
                'Execution': self.df['Execution'] if 'Execution' in self.df.columns else None,
                'Cluster': self.cluster_labels,
                'Text': self.df['original_text'].str[:100] + '...' if 'original_text' in self.df.columns else None
            },
            title='Interactive t-SNE Clustering Visualization',
            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=1000,
            height=700,
            showlegend=True
        )
        
        fig.show()
    
    def analyze_clusters(self, top_n=5):
        """
        Analyze cluster characteristics and show sample statements.
        
        Args:
            top_n (int): Number of sample statements to show per cluster
        """
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS")
        print("="*80)
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            print(f"\n📊 CLUSTER {cluster_id} ({len(cluster_data)} statements)")
            print("-" * 50)
            
            # Show sample statements
            if 'original_text' in cluster_data.columns:
                print("Sample statements:")
                for i, (idx, row) in enumerate(cluster_data.head(top_n).iterrows()):
                    text = row['original_text'][:150] + "..." if len(str(row['original_text'])) > 150 else row['original_text']
                    print(f"  {i+1}. {text}")
            
            # Show execution numbers if available
            if 'Execution' in cluster_data.columns:
                executions = cluster_data['Execution'].tolist()
                print(f"\nExecution numbers: {executions[:10]}{'...' if len(executions) > 10 else ''}")
            
            print()
    
    def save_clustered_data(self, output_file="clustered_statements.csv"):
        """Save the clustered data with cluster labels."""
        if self.cluster_labels is not None:
            self.df.to_csv(output_file, index=False)
            print(f"Clustered data saved to {output_file}")
        else:
            print("No clustering performed yet")
    
    def run_full_analysis(self, n_clusters=8, method='kmeans'):
        """
        Run the complete clustering analysis pipeline.
        
        Args:
            n_clusters (int): Number of clusters
            method (str): Clustering method
        """
        print("🚀 Starting Gemini Clustering Analysis Pipeline")
        print("=" * 60)
        
        # Load and preprocess data
        if not self.load_data():
            return False
        
        if not self.preprocess_embeddings():
            return False
        
        # Find optimal clusters
        # optimal_clusters = self.find_optimal_clusters()
        optimal_clusters = 2 
        print(f"Using {n_clusters} clusters (optimal: {optimal_clusters})")
        
        # Perform clustering
        if not self.perform_clustering(n_clusters, method):
            return False
        
        # Dimensionality reduction
        self.perform_tsne()
        self.perform_pca()
        
        # Visualizations
        self.plot_clusters_tsne()
        self.plot_clusters_pca()
        self.plot_interactive_tsne()
        
        # Analysis
        self.analyze_clusters()
        
        # Save results
        self.save_clustered_data()
        
        print("\n✅ Clustering analysis completed successfully!")
        return True

def main():
    """Main function to run the clustering analysis."""
    
    # Initialize analyzer
    analyzer = GeminiClusteringAnalyzer()
    
    # Run full analysis
    analyzer.run_full_analysis(n_clusters=2, method='kmeans')
    
    # You can also run individual components:
    # analyzer.load_data()
    # analyzer.preprocess_embeddings()
    # analyzer.find_optimal_clusters()
    # analyzer.perform_clustering(n_clusters=6, method='agglomerative')
    # analyzer.perform_tsne()
    # analyzer.plot_clusters_tsne()

if __name__ == "__main__":
    main()

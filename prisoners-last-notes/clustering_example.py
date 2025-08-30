#!/usr/bin/env python3
"""
Example script demonstrating how to use the Gemini Clustering Analyzer.
This script shows different ways to perform clustering analysis on the embeddings.
"""

from clustering import GeminiClusteringAnalyzer
import matplotlib.pyplot as plt

def example_basic_clustering():
    """Basic clustering example with default settings."""
    print("🔍 Example 1: Basic Clustering Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = GeminiClusteringAnalyzer()
    
    # Run full analysis pipeline
    analyzer.run_full_analysis(n_clusters=8, method='kmeans')

def example_custom_clustering():
    """Custom clustering example with different parameters."""
    print("\n🔍 Example 2: Custom Clustering Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = GeminiClusteringAnalyzer()
    
    # Load and preprocess data
    analyzer.load_data()
    analyzer.preprocess_embeddings()
    
    # Find optimal number of clusters
    optimal_clusters = analyzer.find_optimal_clusters(max_clusters=15)
    
    # Try different clustering methods
    methods = ['kmeans', 'agglomerative']
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} clustering ---")
        analyzer.perform_clustering(n_clusters=optimal_clusters, method=method)
        
        # Perform t-SNE and plot
        analyzer.perform_tsne(perplexity=25, n_iter=1500)
        analyzer.plot_clusters_tsne(figsize=(16, 8))
        
        # Analyze clusters
        analyzer.analyze_clusters(top_n=3)
        
        # Save results
        analyzer.save_clustered_data(f"clustered_statements_{method}.csv")

def example_interactive_analysis():
    """Interactive analysis example with custom parameters."""
    print("\n🔍 Example 3: Interactive Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = GeminiClusteringAnalyzer()
    
    # Load and preprocess data
    analyzer.load_data()
    analyzer.preprocess_embeddings()
    
    # Try different numbers of clusters
    cluster_numbers = [6, 8, 10, 12]
    
    for n_clusters in cluster_numbers:
        print(f"\n--- Testing {n_clusters} clusters ---")
        
        # Perform clustering
        analyzer.perform_clustering(n_clusters=n_clusters, method='kmeans')
        
        # Perform t-SNE
        analyzer.perform_tsne(perplexity=30)
        
        # Create interactive plot
        analyzer.plot_interactive_tsne()
        
        # Show cluster distribution
        cluster_counts = analyzer.df['cluster'].value_counts().sort_index()
        print(f"Cluster distribution for {n_clusters} clusters:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} statements")

def example_step_by_step():
    """Step-by-step analysis example."""
    print("\n🔍 Example 4: Step-by-Step Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = GeminiClusteringAnalyzer()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    analyzer.load_data()
    
    # Step 2: Preprocess embeddings
    print("\nStep 2: Preprocessing embeddings...")
    analyzer.preprocess_embeddings()
    
    # Step 3: Find optimal clusters
    print("\nStep 3: Finding optimal number of clusters...")
    optimal_clusters = analyzer.find_optimal_clusters(max_clusters=20)
    
    # Step 4: Perform clustering
    print(f"\nStep 4: Performing clustering with {optimal_clusters} clusters...")
    analyzer.perform_clustering(n_clusters=optimal_clusters, method='kmeans')
    
    # Step 5: Dimensionality reduction
    print("\nStep 5: Performing t-SNE dimensionality reduction...")
    analyzer.perform_tsne(perplexity=30, n_iter=2000)
    
    # Step 6: Visualize results
    print("\nStep 6: Creating visualizations...")
    analyzer.plot_clusters_tsne(figsize=(18, 10))
    
    # Step 7: Analyze clusters
    print("\nStep 7: Analyzing cluster characteristics...")
    analyzer.analyze_clusters(top_n=5)
    
    # Step 8: Save results
    print("\nStep 8: Saving results...")
    analyzer.save_clustered_data("step_by_step_clustering.csv")
    
    print("\n✅ Step-by-step analysis completed!")

def main():
    """Main function to run examples."""
    print("🚀 Gemini Clustering Analysis Examples")
    print("=" * 60)
    print("This script demonstrates different ways to use the clustering analyzer.")
    print("Choose an example to run:\n")
    
    examples = [
        ("Basic Clustering", example_basic_clustering),
        ("Custom Clustering", example_custom_clustering),
        ("Interactive Analysis", example_interactive_analysis),
        ("Step-by-Step Analysis", example_step_by_step)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nRunning all examples...\n")
    
    try:
        # Run all examples
        example_basic_clustering()
        example_custom_clustering()
        example_interactive_analysis()
        example_step_by_step()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have the required data files and dependencies installed.")

if __name__ == "__main__":
    main()

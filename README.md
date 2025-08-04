# Mall Customer Clustering Analysis

This project demonstrates the application of two clustering algorithms (CLARA and DBSCAN) to mall customer data, which includes age, annual income, and spending score metrics.

## File Description

- `main.py`: Main script containing code for data loading, processing, and clustering visualization.
- `Mall_Customers.csv`: Source data file (not included in the repository, must be downloaded separately).

## Dependencies

The project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install them using:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
____
## Clustering Algorithms

1. CLARA (Clustering Large Applications):
   * Based on the k-medoids algorithm.
   * Works with data subsamples to reduce computational complexity.
   * Visualized using a 3D plot where medoids are marked with black crosses.

5. DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
   * Clusters data based on density.
   * Identifies noise points (excluded from visualization).
   * Cluster centroids are marked with black stars.
_____
## Visualization
1. CLARA
<img width="766" height="609" alt="image" src="https://github.com/user-attachments/assets/8605940a-6bc7-47e7-a66a-c33376ea6570" />

2. DBSCAN
<img width="642" height="603" alt="image" src="https://github.com/user-attachments/assets/3081879a-d689-4668-b49a-7411268666a8" />

P.S. you can move graphics 

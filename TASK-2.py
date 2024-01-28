import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime

def display_unique_values(data, selected_features):
    print("\nUnique values for selected features:")
    for feature in selected_features:
        unique_values = data[feature].unique()
        print(f"{feature}: {unique_values}")

def cluster_data(data, selected_features, k):
    numerical_features = data[selected_features]
    categorical_features = [col for col in numerical_features.columns if data[col].dtype == 'object']
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features.columns.difference(categorical_features)),
                                                   ('cat', OneHotEncoder(), categorical_features)])
    X_scaled = preprocessor.fit_transform(numerical_features)
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    return data

file_path = "C:\\Users\\Lenovo\\Desktop\\Internsavy\\Customers.csv"
df = pd.read_csv(file_path)

print("Available features for clustering:", df.columns)

selected_features = input("Enter the features for clustering (comma-separated): ").split(',')


display_unique_values(df, selected_features)


proceed = input("Do you want to proceed with clustering? (yes/no): ").lower()

if proceed == 'yes':
    k = int(input("Enter the number of clusters (k): "))

 
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"C:\\Users\\Lenovo\\Desktop\\Clustered_Customers_{timestamp}.xlsx"

    df = cluster_data(df, selected_features, k)

    print(df[['CustomerID', 'Cluster']])
    
    df.to_excel(output_path, index=False)

    print(f"\nCluster assignments saved to {output_path}")
else:
    print("Clustering process aborted.")

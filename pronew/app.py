import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Title
st.title("🛍 Customer Segmentation Dashboard")

# Load data
df = pd.read_csv('project4.csv')
st.subheader("Raw Data")
st.dataframe(df.head())

# Preprocessing
df.dropna(inplace=True)

# Drop high-cardinality columns
nunique = df.nunique()
df.drop(columns=nunique[nunique > 100].index.tolist(), inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Sidebar for K value
k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(pca_result)

# Add clustering results
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]
df['Cluster'] = clusters

# Silhouette score
sample_size = 5000
if len(df) > sample_size:
    idx = np.random.choice(len(df), sample_size, replace=False)
    score = silhouette_score(pca_result[idx], clusters[idx])
else:
    score = silhouette_score(pca_result, clusters)

st.sidebar.metric("Silhouette Score", f"{score:.2f}")

# PCA scatter plot
st.subheader("Cluster Visualization (PCA + KMeans)")
fig = px.scatter(
    df, x='PC1', y='PC2', color=df['Cluster'].astype(str),
    title='Customer Segments',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig)

# Cluster feature averages
st.subheader("📊 Cluster Feature Averages")
cluster_means = df.groupby('Cluster').mean(numeric_only=True).reset_index()
st.dataframe(cluster_means)

# Bar chart: select features to visualize
st.subheader("📈 Compare Feature Averages Across Clusters (Bar Chart)")
all_features = cluster_means.columns.drop('Cluster').tolist()
selected_features = st.multiselect("Select features to plot", all_features[:5], default=all_features[:2])

if selected_features:
    bar_data = cluster_means.melt(id_vars='Cluster', value_vars=selected_features,
                                   var_name='Feature', value_name='Average')
    bar_fig = px.bar(
        bar_data, x='Feature', y='Average', color=bar_data['Cluster'].astype(str),
        barmode='group',
        title='Average Feature Values by Cluster',
        labels={'Cluster': 'Cluster'}
    )
    st.plotly_chart(bar_fig)
else:
    st.info("Please select at least one feature to display the bar chart.")
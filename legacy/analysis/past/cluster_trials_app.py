import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title='Optuna Trial 聚類分析', layout='wide')
st.title('Optuna Trial 聚類分析工具')
st.markdown('''
本工具針對Optuna trial結果csv進行KMeans聚類，支援降維（PCA），協助你分析策略多樣性與分群。
- 支援上傳csv或選擇現有檔案
- 可選策略/數據源
- 支援KMeans分群（可調分群數）
- 支援PCA降維（2~6維）
- 結果即時可視化
''')

# 1. 選擇或上傳csv
st.header('1. 選擇或上傳Optuna結果csv')
workspace = Path('.')
def find_csv_files():
    files = list(workspace.glob('**/optuna_results_*.csv'))
    return [str(f) for f in files]

csv_files = find_csv_files()
selected_csv = st.selectbox('選擇現有csv檔案', options=['（請選擇）'] + csv_files)
uploaded = st.file_uploader('或上傳csv檔案', type='csv')

if uploaded:
    df = pd.read_csv(uploaded)
    st.success('已載入上傳檔案')
elif selected_csv and selected_csv != '（請選擇）':
    df = pd.read_csv(selected_csv)
    st.success(f'已載入：{selected_csv}')
else:
    st.warning('請先選擇或上傳csv檔案')
    st.stop()

# 2. 選策略/數據源
st.header('2. 選擇策略與數據源')
strategy_options = sorted(df['strategy'].dropna().unique())
data_source_options = sorted(df['data_source'].dropna().unique())
selected_strategy = st.selectbox('策略', options=strategy_options)
selected_data_source = st.selectbox('數據源', options=data_source_options)

filtered_df = df[(df['strategy'] == selected_strategy) & (df['data_source'] == selected_data_source)]
st.write(f'共 {len(filtered_df)} 筆 trial')
test_df = filtered_df.head(200)

# 3. 聚類參數
st.header('3. KMeans分群與降維參數')
n_clusters = st.slider('KMeans分群數', 2, 10, 3)
pca_dim = st.slider('PCA降維維度', 2, min(6, len(test_df.columns)-1, len(test_df)), 2)

# 4. 聚類欄位自動偵測
cluster_cols = ['num_trades', 'sharpe_ratio', 'max_drawdown', 'profit_factor', 'avg_hold_days', 'excess_return_stress']
used_cols = [c for c in cluster_cols if c in test_df.columns and test_df[c].notna().sum() > 0]
if not used_cols:
    st.error('無可用聚類欄位，請確認csv內容')
    st.stop()

# 5. 按鈕觸發聚類
if st.button('執行KMeans聚類分析'):
    st.subheader(f'KMeans聚類+PCA降維結果（前900筆）')
    try:
        X = test_df[used_cols].fillna(test_df[used_cols].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # PCA降維
        pca = PCA(n_components=pca_dim)
        X_pca = pca.fit_transform(X_scaled)
        # KMeans分群
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        clustered = test_df.copy()
        clustered['cluster'] = labels
        for i in range(pca_dim):
            clustered[f'PC{i+1}'] = X_pca[:, i]
        st.write(f'使用欄位: {used_cols}')
        # 2D PCA散點圖
        fig2d = px.scatter(clustered, x='PC1', y='PC2', color='cluster', title='PCA前兩維分群散點圖')
        st.plotly_chart(fig2d, use_container_width=True)
        # scatter matrix
        fig = px.scatter_matrix(clustered, dimensions=used_cols, color='cluster', title='KMeans聚類結果（原始指標空間）')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(clustered[['cluster'] + used_cols + ['score'] + [f'PC{i+1}' for i in range(pca_dim)]])
    except Exception as e:
        st.error(f'KMeans聚類失敗: {e}') 
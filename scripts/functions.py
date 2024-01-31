def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    data_subset = np.asarray(data[max_items, :].todense())

    pca = PCA(n_components=2).fit_transform(data_subset)
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data_subset))

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [plt.cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('t-SNE Cluster Plot')

def plot_3d_pca_tsna(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    data_subset = np.asarray(data[max_items, :].todense())

    pca = PCA(n_components=3).fit_transform(data_subset)
    tsne = TSNE(n_components=3).fit_transform(data_subset)

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [plt.cm.hsv(i / max_label) for i in label_subset[idx]]

    f = plt.figure(figsize=(18, 8))
    
    # 3D PCA Plot
    ax1 = f.add_subplot(121, projection='3d')
    ax1.scatter(pca[idx, 0], pca[idx, 1], pca[idx, 2], c=label_subset)
    ax1.set_title('3D PCA Cluster Plot')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_zlabel('Principal Component 3')

    # 3D t-SNE Plot
    ax2 = f.add_subplot(122, projection='3d')
    ax2.scatter(tsne[idx, 0], tsne[idx, 1], tsne[idx, 2], c=label_subset)
    ax2.set_title('3D t-SNE Cluster Plot')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_zlabel('t-SNE Dimension 3')

    plt.show()

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    silhouette_scores = []
    
    for k in iters:
        kmeans = MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20)
        kmeans.fit(data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        
        print(f'Fit {k} clusters - Silhouette Score: {silhouette_avg}')
        
    # Plot silhouette scores
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, silhouette_scores, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score by Cluster Center Plot')

    
def get_top_keywords(data, clusters, feature_names, n_terms):
    df = pd.DataFrame(np.asarray(data.todense())).groupby(clusters).mean()
    
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([feature_names[t] for t in np.argsort(r)[-n_terms:]]))

def plot_tsne_pca_emb(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    data_subset = np.asarray(data[max_items, :])

    n_components_pca = min(data_subset.shape[0], data_subset.shape[1])  # Adjusted line
    pca = PCA(n_components=n_components_pca).fit_transform(data_subset)
    
    tsne = TSNE().fit_transform(pca)  # Use PCA output for t-SNE

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [plt.cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('t-SNE Cluster Plot')

def plot_3d_pca_tsna_emb(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(len(data)), size=3000, replace=False)

    data_subset = np.asarray(data[max_items, :])

    # 3D PCA
    pca = PCA(n_components=3).fit_transform(data_subset)

    # 3D t-SNE
    tsne = TSNE(n_components=3).fit_transform(data_subset)

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [plt.cm.hsv(i / max_label) for i in label_subset[idx]]

    f = plt.figure(figsize=(18, 8))

    # 3D PCA Plot
    ax1 = f.add_subplot(121, projection='3d')
    ax1.scatter(pca[idx, 0], pca[idx, 1], pca[idx, 2], c=label_subset)
    ax1.set_title('3D PCA Cluster Plot')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_zlabel('Principal Component 3')

    # 3D t-SNE Plot
    ax2 = f.add_subplot(122, projection='3d')
    ax2.scatter(tsne[idx, 0], tsne[idx, 1], tsne[idx, 2], c=label_subset)
    ax2.set_title('3D t-SNE Cluster Plot')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_zlabel('t-SNE Dimension 3')

    plt.show()
    
def find_optimal_clusters_with_embeddings(tokenized_text, max_k):
    # Train Word2Vec model
    word2vec_model = word2vec.Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

    # Get embeddings for each document
    embeddings = [np.mean([word2vec_model.wv[word] for word in doc], axis=0) for doc in tokenized_text]

    # Standardize the embeddings
    data_standardized = StandardScaler().fit_transform(embeddings)

    iters = range(2, max_k+1, 2)
    
    silhouette_scores = []
    
    for k in iters:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=20)
        kmeans.fit(data_standardized)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(data_standardized, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        
        print(f'Fit {k} clusters - Silhouette Score: {silhouette_avg}')
        
    # Plot silhouette scores
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, silhouette_scores, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score by Cluster Center Plot')
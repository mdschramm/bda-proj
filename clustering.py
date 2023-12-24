import os.path
import numpy as np
import pandas as pd
from process_data import X, tracks, track_features, cos_sim, SESSION_ID, session_groups, nearest_tracks, N_TESTS_PER_USER
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pickle
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity

# Cluster training tracks by features - build map from track_id to cluster

N_TRACK_CLUSTERS = 3000
N_USER_CLUSTERS = 100
COLLAB_FILTERING_NEIGHBORS = 8
CLUSTER_CENTERS_FILE = 'track_feature_cluster_centers.npy'
CLUSTER_LABELS_FILE = 'track_feature_cluster_labels.npy'


def cluster_tracks_by_features(tracks, n_clusters=N_TRACK_CLUSTERS):
    if os.path.isfile(CLUSTER_CENTERS_FILE) and os.path.isfile(CLUSTER_LABELS_FILE):
        centers, labels = np.load(
            CLUSTER_CENTERS_FILE), np.load(CLUSTER_LABELS_FILE)
        return centers, labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=123,
                    n_init='auto').fit(tracks)

    np.save(CLUSTER_CENTERS_FILE, kmeans.cluster_centers_)
    np.save(CLUSTER_LABELS_FILE, kmeans.labels_)
    return kmeans.cluster_centers_, kmeans.labels_


TRACK_CLUSTER = 'TRACK_CLUSTER'

# Use training rows of X from session_groups_train and track cluster map to create user preferences matrix
# - do additively from multiple tracks liked in same cluster
SESSION_IDX_COL = 'session_idx'
unique_sessions = X[SESSION_ID].unique()


def add_session_indexes(X):

    X[SESSION_IDX_COL] = X[SESSION_ID].apply(
        lambda x: list(unique_sessions).index(x))
    return X


X_EXTRA_COLS = 'x_xtra_cols.pkl'
TRACKS_WITH_TC_FILE = 'tracks_tc.pkl'


def add_session_i_and_tc_to_X(X):
    if os.path.isfile(X_EXTRA_COLS) and os.path.isfile(TRACKS_WITH_TC_FILE):
        with open(X_EXTRA_COLS, 'rb') as f, open(TRACKS_WITH_TC_FILE, 'rb') as g:
            return pickle.load(f), pickle.load(g)
    X = add_session_indexes(X)
    tc_centers, labels = cluster_tracks_by_features(tracks[track_features])
    tracks[TRACK_CLUSTER] = labels
    X = pd.merge(X, tracks[['track_id', TRACK_CLUSTER]],
                 left_on='track_id_clean', right_on='track_id', how='left', sort=False)
    with open(X_EXTRA_COLS, 'wb') as f, open(TRACKS_WITH_TC_FILE, 'wb') as g:
        pickle.dump(X, f)
        pickle.dump(tracks, g)
    return X, tracks


X, tracks = add_session_i_and_tc_to_X(X)


def get_user_track_cluster_pref(train_groups):
    user_track_clusters = np.zeros((len(unique_sessions), N_TRACK_CLUSTERS))
    train_users = [key for d in train_groups for key in d.keys()]
    session_cluster_pairs = X[X[SESSION_ID].isin(
        train_users)][[SESSION_IDX_COL, TRACK_CLUSTER]]

    def process_row(row):
        user_track_clusters[row[SESSION_IDX_COL], row[TRACK_CLUSTER]] += 1

    session_cluster_pairs.apply(process_row, axis=1)

    non_zero_rows = user_track_clusters[~np.all(
        user_track_clusters == 0, axis=1)]
    # delete zero rows for users that were not in training set
    return non_zero_rows

# groups users by their vector of track cluster preferences


def get_uc_to_tc(user_tc):
    kmeans = KMeans(n_clusters=N_USER_CLUSTERS, random_state=123,
                    n_init='auto').fit(user_tc)
    uc_to_tc = np.zeros((N_USER_CLUSTERS, N_TRACK_CLUSTERS))

    # add assigned cluster for each user
    for i, row in enumerate(user_tc):
        uc = kmeans.labels_[i]
        uc_to_tc[uc] += row
    return uc_to_tc, kmeans


def get_group_features_and_labels(group):
    user_features = []
    labels = []
    for session_id in group:
        rows = group[session_id]
        x_rows = rows[:-N_TESTS_PER_USER]
        y_rows = rows[-N_TESTS_PER_USER:]
        user_features.append(x_rows)
        labels += y_rows

    return user_features, X[track_features].iloc[labels].to_numpy()


def get_track_cluster_vectors(user_features, X):
    tc_vectors = []
    for user in user_features:
        user_tc_vector = np.zeros(N_TRACK_CLUSTERS)
        user_x_clusters = X.iloc[user][TRACK_CLUSTER].to_numpy()
        for cluster in user_x_clusters:
            user_tc_vector[cluster] += 1
        tc_vectors.append(user_tc_vector)
    return tc_vectors
# The collaborative filtering part

# Looks for track cluster that all user's User clusters like


def find_tc_overlap(user_clusters, uc_to_tc, uc_neighbors):
    user_vecs = uc_to_tc[user_clusters]
    neighbors = uc_neighbors.kneighbors(
        user_vecs, n_neighbors=COLLAB_FILTERING_NEIGHBORS, return_distance=False)

    overlap_products = np.prod(user_vecs, axis=0)
    max_overlap_prod = np.max(overlap_products)
    n = 0
    while (max_overlap_prod == 0 and n < COLLAB_FILTERING_NEIGHBORS):
        neighbor_vecs = uc_to_tc[np.transpose(neighbors[:, n])]
        user_vecs = np.add(user_vecs, neighbor_vecs)
        overlap_products = np.prod(user_vecs, axis=0)
        max_overlap_prod = np.max(overlap_products)
        if n == COLLAB_FILTERING_NEIGHBORS:
            print('no full overlap')
            break
        n += 1
    column_sums = np.sum(user_vecs, axis=0)
    best_tc = np.argmax(
        overlap_products) if max_overlap_prod > 0 else np.argmax(column_sums)  # fallback on sum if no consensus
    return best_tc


def find_track_cluster_overlap(user_features, uc_to_tc, kmeans, X, uc_neighbors):
    track_cluster_vectors = get_track_cluster_vectors(user_features, X)
    user_clusters = kmeans.predict(track_cluster_vectors)
    return find_tc_overlap(user_clusters, uc_to_tc, uc_neighbors)

# limits baseline search over average to just search over cluster
# makes sense because this minimizes impact of outliers


def find_nearest_track_in_cluster(tc, user_features, X):
    x_rows = reduce(lambda a, b: a + b, user_features)
    user_tracks = X[track_features].iloc[x_rows].to_numpy()
    avg = np.array([np.mean(user_tracks, axis=0)])
    cluster_tracks = tracks[tracks[TRACK_CLUSTER]
                            == tc][track_features].to_numpy()
    cosine_similarities = cosine_similarity(avg.reshape(1, -1), cluster_tracks)
    min_sim_index = np.argmin(cosine_similarities)

    return cluster_tracks[min_sim_index]


def get_split_sims(test_groups, uc_to_tc, kmeans, X, uc_neighbors):
    total_cos_sim = 0
    total_y_rows = 0
    for group in test_groups:
        user_features, y_rows = get_group_features_and_labels(group)
        total_y_rows += len(y_rows)
        tc = find_track_cluster_overlap(
            user_features, uc_to_tc, kmeans, X, uc_neighbors)
        pred = find_nearest_track_in_cluster(tc, user_features, X)

        total_cos_sim += sum([cos_sim(pred, row)
                              for row in y_rows])

    return total_cos_sim, total_y_rows


# create 10 kfold splits of .9 train and .1 test
kf = KFold(n_splits=10)
group_splits = kf.split(session_groups)

total_sim = 0
total_rows = 0
for g_train, g_test in group_splits:
    train_groups = [session_groups[i] for i in g_train]
    # .0065 fraction non-zero entries
    user_track_cluster_pref = get_user_track_cluster_pref(train_groups)
    uc_to_tc, kmeans = get_uc_to_tc(user_track_cluster_pref)
    # .117 fraction non-zero still too sparse for kd-tree
    uc_neighbors = NearestNeighbors(
        n_neighbors=COLLAB_FILTERING_NEIGHBORS, metric='cosine')
    uc_neighbors.fit(uc_to_tc)
    test_groups = [session_groups[i] for i in g_test]
    sim_sum, n_rows = get_split_sims(
        test_groups, uc_to_tc, kmeans, X, uc_neighbors)
    total_sim += sim_sum
    total_rows += n_rows

print(total_rows)
print(total_sim / total_rows)

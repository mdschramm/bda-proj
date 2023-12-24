import json
import os.path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, normalize, StandardScaler
from category_encoders import CountEncoder
from sklearn.neighbors import NearestNeighbors


rng = np.random.default_rng(seed=123)

log_mini = 'data/training_set/log_mini.csv'
tf_mini = 'data/track_features/tf_mini.csv'

sessions = pd.read_csv(log_mini)
tracks = pd.read_csv(tf_mini)

sessions_join_index = 'track_id_clean'
track_features_index = 'track_id'

data = pd.merge(sessions, tracks,
                left_on='track_id_clean', right_on='track_id')

N_TESTS_PER_USER = 4
MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 30

'''
Labeling:

LIKE:
- if skip_3 or not_skipped or 
- hist_user_behavior_n_seekfwd or hist_user_behavior_n_seekback = True or context_type = usercollection

otherwise DISLIKE

Pre-processing features:
- one-hot-encode: key, mode, time-signature
- everything else StandardScaled
'''

LABEL_COL = 'liked'
SESSION_ID = 'session_id'


def add_labels(data):
    data['usercollection'] = np.where(
        data['context_type'] == 'usercollection', True, False)
    data[LABEL_COL] = data['skip_3'] | data['not_skipped'] | data['hist_user_behavior_n_seekfwd'] | data[
        'hist_user_behavior_n_seekback'] | data['usercollection']
    return data


'''
Drop unliked listening instances
Drop sessions length 1
'''


def drop_unliked_sessions(data):
    data = data.drop(index=data[data[LABEL_COL] == False].index)
    session_like_counts = Counter(data[SESSION_ID].tolist())
    liked_sessions = []
    total_like_count = 0
    for id in session_like_counts:
        total_like_count += session_like_counts[id]
        if session_like_counts[id] > N_TESTS_PER_USER:
            liked_sessions.append(id)

    data = data[data[SESSION_ID].isin(liked_sessions)]
    data = data.reset_index(drop=True)
    return data, liked_sessions


'''
Shuffle session_ids and create random groups of 2-10 sessions
Map each group to list of row indexes
'''

SESSION_GROUPS_FILE = 'session_groups.json'


def add_to_session_groups(data, session_ids, session_groups):
    group_map = {}
    for id in session_ids:
        group_map[id] = data[data[SESSION_ID] == id].index.tolist()
    session_groups.append(group_map)


def get_session_groups(data, session_ids):
    if os.path.isfile(SESSION_GROUPS_FILE):
        with open(SESSION_GROUPS_FILE) as f:
            session_groups = json.load(f)
            return session_groups

    n_sessions = len(session_ids)
    rng.shuffle(session_ids)
    sessions_seen = 0
    session_groups = []
    while sessions_seen < n_sessions - MAX_GROUP_SIZE:
        group_size = rng.integers(MIN_GROUP_SIZE, MAX_GROUP_SIZE + 1)
        new_seen = sessions_seen + group_size
        add_to_session_groups(
            data, session_ids[sessions_seen:new_seen], session_groups)
        sessions_seen = new_seen
    add_to_session_groups(
        data, session_ids[sessions_seen:], session_groups)
    with open(SESSION_GROUPS_FILE, 'w') as f:
        json.dump(session_groups, f)
    return session_groups


track_features = [
    'duration',
    'release_year',
    'us_popularity_estimate',
    'acousticness',
    'beat_strength',
    'bounciness',
    'danceability',
    'dyn_range_mean',
    'energy',
    'flatness',
    'instrumentalness',
    'key',
    'liveness',
    'loudness',
    'mechanism',
    'mode',
    'organism',
    'speechiness',
    'tempo',
    'time_signature',
    'valence',
    'acoustic_vector_0',
    'acoustic_vector_1',
    'acoustic_vector_2',
    'acoustic_vector_3',
    'acoustic_vector_4',
    'acoustic_vector_5',
    'acoustic_vector_6',
    'acoustic_vector_7',
]

# StandardScaling and one-hot encoding


def process_numerics(X, y=None):
    column_transformer = ColumnTransformer(
        [
            # normalize numerical features
            ('scale_numbers', StandardScaler(),
             make_column_selector(dtype_include=np.float64)),
            # normalize release year
            ('scale_year', StandardScaler(), ['release_year']),
            # one-hot encode the categorical variables
            ('encode_major_or_minor', OrdinalEncoder(),
             ['mode']),
            ('target_encode_key_mode', CountEncoder(
                normalize=True, cols=['key', 'time_signature']), ['key', 'time_signature'])
        ], remainder='passthrough'
    )
    X.loc[:, track_features] = column_transformer.fit_transform(
        X.loc[:, track_features], y)

    X.loc[:, track_features] = normalize(X.loc[:, track_features])
    return X


labels_added = add_labels(data)
unlike_dropped, session_ids = drop_unliked_sessions(
    labels_added)
session_groups = get_session_groups(unlike_dropped, session_ids)

X = process_numerics(unlike_dropped)
tracks = process_numerics(tracks)

# n_neighbors=5 is just an initializer optimization paramter for expected queries
nearest_tracks = NearestNeighbors(
    n_neighbors=5, metric='cosine').fit(tracks[track_features].to_numpy())


def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

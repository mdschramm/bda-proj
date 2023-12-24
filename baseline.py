from process_data import rng, X, tracks, track_features, session_groups, nearest_tracks, cos_sim, N_TESTS_PER_USER
from sklearn.preprocessing import normalize
import numpy as np

# test random point
p = np.array(
    normalize(np.array([rng.random(tracks[track_features].shape[1])])))
# dist, ind = nearest_tracks.query(
#     p, k=10)
# print(ind)
# print((cos_sim(tracks[track_features].iloc[ind[0]], np.transpose(p))))

'''Baseline model:

- For each session-group:
---- For each user, hold out a song that they like
---- Average all the others and find nearest song to average
---- Calculate average distance between recommended song and hold-out songs'''


'''
Splits group into training and test sessions
'''

# TODO could create more train-test instances


def split_session_group(group):
    x_rows, y_rows = [], []
    for session_id in group:
        rows = group[session_id]
        x_rows += rows[:-N_TESTS_PER_USER]
        y_rows += rows[-N_TESTS_PER_USER:]

    return x_rows, y_rows


def get_instances(groups):
    return list(map(lambda g: split_session_group(g), groups))


# avg cos-sim: 0.19861769
def run_baseline(X, tracks, groups):
    instances = get_instances(groups)
    total_y = 0
    total_x = 0
    total_cos_sim = 0
    for (x, y) in instances:
        x_rows = (X.iloc[x]).to_numpy()
        y_rows = (X.iloc[y]).to_numpy()
        avg = np.array([np.mean(x_rows, axis=0)])
        ind = nearest_tracks.kneighbors(
            avg, n_neighbors=1, return_distance=False)
        # pred = tracks.iloc[[ind[0][0]]]  # 0.16990661207465263
        pred = np.mean(y_rows, axis=0)  # .281
        # pred = tracks.iloc[0]  # -0.009710992261819977
        total_x += x_rows.shape[0]
        total_y += y_rows.shape[0]
        total_cos_sim += sum([cos_sim(pred, row)
                              for row in y_rows])
    return total_cos_sim / total_y


avg_sim = run_baseline(
    X[track_features], tracks[track_features], session_groups)
print(avg_sim)

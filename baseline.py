from process_data import rng, X, session_groups, kd_tree
import numpy as np

# test random point
# dist, ind = kd_tree.query(np.array([rng.random(X.shape[1])]), k=10)
# [[2.58619351 2.58619351 2.58619351 2.58619351 2.58619351 2.58619351 2.58619351 2.58619351 2.58619351 2.58619351]]
# print(dist)

'''Baseline model:

- For each session-group:
---- For each user, hold out a song that they like
---- Average all the others and find nearest song to average
---- Calculate average distance between recommended song and hold-out songs'''


def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


'''
Splits group into training and test sessions
'''


def split_session_group(group):
    train_rows, test_rows = [], []
    for session_id in group:
        rows = group[session_id]
        train_rows += rows[:-1]
        test_rows += rows[-1:]

    return train_rows, test_rows


def get_instances(groups):
    return list(map(lambda g: split_session_group(g), groups))


def run_baseline(X, groups):
    instances = get_instances(groups)
    total_tests = 0
    total_cos_sim = np.zeros((1, X.shape[1]))
    for (train, test) in instances:
        train_rows = X[train, :]
        test_rows = X[test, :]
        avg = np.array([np.mean(train_rows, axis=0)])
        dist_avg_to_pred, ind = kd_tree.query(avg, k=1)
        pred = X[ind]
        total_tests += len(test_rows)
        total_cos_sim += sum([cos_sim(pred, row)
                              for row in test_rows])
    print(X.shape, print(total_tests))
    return total_cos_sim / total_tests


avg_sim = run_baseline(X, session_groups)
print(avg_sim)

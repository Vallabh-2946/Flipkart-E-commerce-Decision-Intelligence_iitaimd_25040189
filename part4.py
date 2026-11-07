"""Part 4: structured model comparison, boosting arithmetic, and sizing."""
import math
import numpy as np

from common import write_csv, write_results
from part2 import confusion, get_split_data, knn_probabilities, metrics, sigmoid


def entropy(y):
    if len(y) == 0:
        return 0.0
    p = float(np.mean(y))
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def gini(y):
    if len(y) == 0:
        return 0.0
    p = float(np.mean(y))
    return 1.0 - p * p - (1.0 - p) ** 2


def gain(y, yes, impurity):
    no = ~yes
    return impurity(y) - (np.sum(yes) / len(y)) * impurity(y[yes]) - (np.sum(no) / len(y)) * impurity(y[no])


def main():
    names, numeric, y, train_idx, test_idx, standardized = get_split_data()
    train_y, test_y = y[train_idx], y[test_idx]
    train_x, test_x = standardized[train_idx], standardized[test_idx]
    raw = numeric
    # The four requested tests use original units, and raw columns are X1..X7.
    raw_x = raw[:, :7]
    split_masks = [
        raw_x[train_idx, 5] == 1,
        raw_x[train_idx, 6] == 1,
        raw_x[train_idx, 2] > 0.5,
        raw_x[train_idx, 0] <= 24,
    ]
    split_names = ["S1", "S2", "S3", "S4"]
    root_h = entropy(train_y)
    root_g = gini(train_y)
    split_rows = []
    info_gains = []
    gini_gains = []
    for name, mask in zip(split_names, split_masks):
        yes_y = train_y[mask]
        no_y = train_y[~mask]
        info_gain = gain(train_y, mask, entropy)
        gin_gain = gain(train_y, mask, gini)
        info_gains.append(info_gain)
        gini_gains.append(gin_gain)
        split_rows.append([name, len(yes_y), int(np.sum(yes_y)), len(no_y), int(np.sum(no_y)), f"{info_gain:.4f}", f"{gin_gain:.4f}"])
    best_info = split_names[int(np.argmax(info_gains))]
    best_gini = split_names[int(np.argmax(gini_gains))]
    write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], split_rows)

    split_index = split_names.index(best_info)
    stump_mask_train = split_masks[split_index]
    yes_rate = float(np.mean(train_y[stump_mask_train]))
    no_rate = float(np.mean(train_y[~stump_mask_train]))
    if split_index == 0:
        stump_yes_test = raw_x[test_idx, 5] == 1
    elif split_index == 1:
        stump_yes_test = raw_x[test_idx, 6] == 1
    elif split_index == 2:
        stump_yes_test = raw_x[test_idx, 2] > 0.5
    else:
        stump_yes_test = raw_x[test_idx, 0] <= 24
    stump_prob = np.where(stump_yes_test, yes_rate, no_rate)

    # Refit the pinned Task 2 logistic model and KNN model from the training data.
    from part2 import fit_logistic
    weights, bias, _ = fit_logistic(train_x, train_y, 0.1, 1000)
    logistic_prob = sigmoid(test_x @ weights + bias)
    k_values = list(range(1, 22, 2))
    cv = {}
    positions = np.arange(len(train_idx))
    for k in k_values:
        scores = []
        for fold in range(5):
            valid = positions % 5 == fold
            probability, _ = knn_probabilities(train_x[~valid], train_y[~valid], train_x[valid], [k])
            scores.append(metrics(train_y[valid], (probability[k] >= 0.5).astype(int))["f1"])
        cv[k] = float(np.mean(scores))
    best_k = min(k for k in k_values if cv[k] == max(cv.values()))
    knn_prob, _ = knn_probabilities(train_x, train_y, test_x, [best_k])
    knn_probability = knn_prob[best_k]
    probabilities = [logistic_prob, knn_probability, stump_prob]
    hard_predictions = [(prob >= 0.3).astype(int) for prob in probabilities]
    models = [
        ("logistic", hard_predictions[0]),
        ("knn", hard_predictions[1]),
        ("stump", hard_predictions[2]),
        ("hard_vote", (np.sum(hard_predictions, axis=0) >= 2).astype(int)),
        ("soft_vote", (np.mean(probabilities, axis=0) >= 0.3).astype(int)),
    ]
    voting_rows = []
    for name, prediction in models:
        m = metrics(test_y, prediction)
        voting_rows.append([name, m["tp"], m["fp"], m["tn"], m["fn"],
                            f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"])
    write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], voting_rows)

    y_boost = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=float)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=float)
    weights_boost = np.full(8, 1.0 / 8.0)
    boost_rows = []
    errors = []
    alphas = []
    for round_no, prediction in ((1, h1), (2, h2)):
        before = weights_boost.copy()
        error = float(np.sum(before[prediction != y_boost]))
        alpha = 0.5 * math.log((1.0 - error) / error)
        unnormalised = before * np.exp(-alpha * y_boost * prediction)
        after = unnormalised / np.sum(unnormalised)
        boost_rows.extend([[round_no, i + 1, f"{before[i]:.4f}", f"{unnormalised[i]:.4f}", f"{after[i]:.4f}"] for i in range(8)])
        errors.append(error)
        alphas.append(alpha)
        weights_boost = after
    write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], boost_rows)
    ensemble = np.sign(alphas[0] * h1 + alphas[1] * h2)
    ensemble_accuracy = float(np.mean(ensemble == y_boost))

    # Image-sized parameter counts, with convolution and pooling shapes tracked explicitly.
    architecture = [
        ("mlp_hidden1", 784 * 16 + 16),
        ("mlp_hidden2", 16 * 16 + 16),
        ("mlp_output", 16 * 10 + 10),
        ("cnn_conv1", (5 * 5 * 1 + 1) * 6),
        ("cnn_conv2", (5 * 5 * 6 + 1) * 16),
        ("cnn_dense1", (5 * 5 * 16) * 32 + 32),
        ("cnn_dense2", 32 * 10 + 10),
    ]
    write_csv("part4_architecture.csv", ["layer", "parameters"], architecture)
    mlp_total = sum(value for name, value in architecture[:3])
    cnn_total = sum(value for name, value in architecture[3:])
    fc_layer_params = (128 * 128 * 3) * 1024 + 1024
    conv_layer_params = (3 * 3 * 3) * 64 + 64
    rnn_total = 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16

    results = [
        ("root_entropy", root_h), ("root_gini", root_g),
        ("best_split_by_info_gain", best_info), ("best_split_by_gini_gain", best_gini),
        ("stump_rate_yes", yes_rate), ("stump_rate_no", no_rate),
        ("round1_error", errors[0]), ("round1_alpha", alphas[0]),
        ("round2_error", errors[1]), ("round2_alpha", alphas[1]),
        ("ensemble_accuracy", ensemble_accuracy), ("oob_fraction", (1.0 - 1.0 / 560) ** 560),
        ("mlp_total", mlp_total), ("cnn_total", cnn_total),
        ("fc_layer_params", fc_layer_params), ("conv_layer_params", conv_layer_params),
        ("rnn_total", rnn_total),
    ]
    write_results("part4_results.txt", results)


if __name__ == "__main__":
    main()

"""Part 2: logistic regression and K-nearest neighbours from scratch."""
import numpy as np

from common import load_data, write_csv, write_results


def stratified_split(y):
    train_indices = []
    test_indices = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        cutoff = (7 * len(indices)) // 10
        train_indices.extend(indices[:cutoff])
        test_indices.extend(indices[cutoff:])
    return np.array(sorted(train_indices)), np.array(sorted(test_indices))


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, probabilities):
    p = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_logistic(x, y, learning_rate, updates, record_epochs=()):
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    recorded = {}
    if 0 in record_epochs:
        recorded[0] = bce(y, sigmoid(x @ weights + bias))
    for epoch in range(1, updates + 1):
        probabilities = sigmoid(x @ weights + bias)
        error = probabilities - y
        weights -= learning_rate * (x.T @ error) / len(y)
        bias -= learning_rate * float(np.mean(error))
        if epoch in record_epochs:
            recorded[epoch] = bce(y, sigmoid(x @ weights + bias))
    return weights, bias, recorded


def confusion(y, pred):
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tp, fp, tn, fn


def metrics(y, pred):
    tp, fp, tn, fn = confusion(y, pred)
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1, "specificity": specificity}


def knn_probabilities(reference_x, reference_y, query_x, k_values):
    squared = np.sum((query_x[:, None, :] - reference_x[None, :, :]) ** 2, axis=2)
    order = np.argsort(squared, axis=1, kind="stable")
    max_k = max(k_values)
    nearest_y = reference_y[order[:, :max_k]]
    cumulative = np.cumsum(nearest_y, axis=1)
    return {k: cumulative[:, k - 1] / k for k in k_values}, nearest_y


def get_split_data():
    names, _, numeric = load_data()
    y = numeric[:, 7].astype(int)
    train_idx, test_idx = stratified_split(y)
    x = numeric[:, :7]
    means = np.mean(x[train_idx], axis=0)
    sds = np.std(x[train_idx], axis=0, ddof=1)
    standardized = (x - means) / sds
    return names, numeric, y, train_idx, test_idx, standardized


def main():
    _, numeric, y, train_idx, test_idx, standardized = get_split_data()
    train_x, train_y = standardized[train_idx], y[train_idx]
    test_x, test_y = standardized[test_idx], y[test_idx]

    weights, bias, losses = fit_logistic(train_x, train_y, 0.1, 1000, (0, 1, 10, 100, 1000))
    test_prob = sigmoid(test_x @ weights + bias)
    train_prob = sigmoid(train_x @ weights + bias)
    results = [
        ("train_rows", len(train_idx)), ("train_positives", int(np.sum(train_y))),
        ("test_rows", len(test_idx)), ("test_positives", int(np.sum(test_y))),
    ]
    results.extend((f"loss_epoch_{epoch}", losses[epoch]) for epoch in (0, 1, 10, 100, 1000))
    results.extend((f"weight_X{i+1}", weights[i]) for i in range(7))
    results.extend([("bias", bias), ("test_bce", bce(test_y, test_prob))])

    for eta in (0.01, 0.1, 1.0):
        _, _, rate_losses = fit_logistic(train_x, train_y, eta, 200, (200,))
        results.append((f"loss_epoch_200_eta_{eta}", rate_losses[200]))
    results.extend((f"odds_ratio_X{i+1}", np.exp(weights[i])) for i in range(7))

    for tau in (0.5, 0.3):
        m = metrics(test_y, (test_prob >= tau).astype(int))
        for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"):
            results.append((f"tau_{tau}_{key}", m[key]))
    results.append(("blind_accuracy", float(np.mean(test_y == 0))))

    threshold_rows = []
    for tau in np.arange(0.1, 1.0, 0.1):
        tp, fp, tn, fn = confusion(test_y, (test_prob >= tau).astype(int))
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        threshold_rows.append([f"{tau:.1f}", tp, fp, tn, fn, f"{tpr:.4f}", f"{fpr:.4f}"])
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], threshold_rows)

    k_values = list(range(1, 22, 2))
    cv_rows = []
    cv_scores = {}
    positions = np.arange(len(train_idx))
    for k in k_values:
        fold_f1 = []
        for fold in range(5):
            valid = positions % 5 == fold
            reference = ~valid
            probabilities, _ = knn_probabilities(train_x[reference], train_y[reference], train_x[valid], [k])
            prediction = (probabilities[k] >= 0.5).astype(int)
            fold_f1.append(metrics(train_y[valid], prediction)["f1"])
        cv_scores[k] = float(np.mean(fold_f1))
        cv_rows.append([k, f"{cv_scores[k]:.4f}"])
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_rows)
    best_k = min(k for k in k_values if cv_scores[k] == max(cv_scores.values()))
    knn_probs, _ = knn_probabilities(train_x, train_y, test_x, [best_k])
    knn_metrics = metrics(test_y, (knn_probs[best_k] >= 0.5).astype(int))
    results.append(("knn_best_k", best_k))
    results.extend((f"knn_{key}", knn_metrics[key]) for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1"))

    train_costs = []
    for k in range(1, 20):
        tau = k / 20
        pred = (train_prob >= tau).astype(int)
        _, fp, _, fn = confusion(train_y, pred)
        train_costs.append((50 * fp + 400 * fn, tau))
    min_cost = min(cost for cost, _ in train_costs)
    cost_tau = min(tau for cost, tau in train_costs if cost == min_cost)
    test_at_tau = confusion(test_y, (test_prob >= cost_tau).astype(int))
    test_at_half = confusion(test_y, (test_prob >= 0.5).astype(int))
    cost_at_tau = 50 * test_at_tau[1] + 400 * test_at_tau[3]
    cost_at_half = 50 * test_at_half[1] + 400 * test_at_half[3]
    results.extend([
        ("cost_tau", cost_tau), ("cost_train_at_tau", min_cost),
        ("cost_test_at_tau", cost_at_tau), ("cost_test_at_0.5", cost_at_half),
        ("cost_test_flag_nobody", 400 * int(np.sum(test_y == 1))),
    ])
    write_results("part2_results.txt", results)


if __name__ == "__main__":
    main()

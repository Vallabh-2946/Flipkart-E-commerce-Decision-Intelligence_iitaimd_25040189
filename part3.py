"""Part 3: k-means, agglomerative clustering, and PCA."""
import numpy as np

from common import load_data, write_csv, write_results


def kmeans(x, k):
    step = len(x) // k
    centers = x[np.arange(k) * step].copy()
    labels = np.full(len(x), -1, dtype=int)
    while True:
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        unchanged = np.array_equal(new_labels, labels)
        labels = new_labels
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                centers[cluster] = np.mean(members, axis=0)
        if unchanged:
            break
    final_sq = np.sum((x - centers[labels]) ** 2, axis=1)
    return labels, centers, float(np.sum(final_sq))


def cluster_distance(a, b, matrix, method):
    values = matrix[np.ix_(a, b)].ravel()
    if method == "single":
        return float(np.min(values))
    if method == "complete":
        return float(np.max(values))
    return float(np.mean(values))


def agglomerate(matrix, method):
    clusters = [[i] for i in range(len(matrix))]
    history = []
    for merge_no in range(1, len(matrix)):
        candidates = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = cluster_distance(clusters[i], clusters[j], matrix, method)
                candidates.append((distance, min(clusters[i]), min(clusters[j]), i, j))
        distance, _, _, i, j = min(candidates)
        a, b = sorted((clusters[i], clusters[j]), key=lambda group: min(group))
        label_a = "+".join(f"E{idx + 1}" for idx in a)
        label_b = "+".join(f"E{idx + 1}" for idx in b)
        history.append((method, merge_no, label_a, label_b, distance))
        merged = sorted(clusters[i] + clusters[j])
        clusters = [group for idx, group in enumerate(clusters) if idx not in (i, j)]
        clusters.append(merged)
        clusters.sort(key=lambda group: min(group))
    heights = [row[4] for row in history]
    gaps = np.diff(heights)
    cut_gap = int(np.argmax(gaps))
    merges_below = cut_gap + 1
    cut_clusters = 6 - merges_below
    return history, cut_clusters


def main():
    names, _, data = load_data()
    original = data[:, :5]
    mean = np.mean(original, axis=0)
    sd = np.std(original, axis=0, ddof=1)
    x = (original - mean) / sd
    all_labels = {}
    wcss = {}
    results = []
    for k in range(2, 7):
        labels, _, value = kmeans(x, k)
        all_labels[k] = labels
        wcss[k] = value
        results.append((f"wcss_k{k}", value))
    elbow = next((k for k in (3, 4, 5)
                  if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k])), 0)
    results.append(("elbow_k", elbow))

    labels3 = all_labels[3]
    order = sorted(range(3), key=lambda cluster: float(np.mean(original[labels3 == cluster, 2])))
    segment_ids = np.empty(len(x), dtype=int)
    segment_rows = []
    segment_names = []
    for ordinal, cluster in enumerate(order):
        name = chr(ord("A") + ordinal)
        members = labels3 == cluster
        segment_ids[members] = ordinal
        segment_names.append(name)
        rows = data[members]
        row = [name, int(np.sum(members))]
        row.extend(f"{value:.4f}" for value in np.mean(rows[:, :5], axis=0))
        row.extend([f"{np.mean(rows[:, 7]):.4f}", f"{np.mean(rows[:, 8]):.4f}"])
        segment_rows.append(row)
        results.append((f"size_segment_{name}", int(np.sum(members))))
    write_csv("part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
              segment_rows)

    distances = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ])
    hierarchical_rows = []
    for method in ("single", "complete", "average"):
        history, count = agglomerate(distances, method)
        hierarchical_rows.extend([[*row[:4], f"{row[4]:.4f}"] for row in history])
        results.append((f"{method}_clusters_at_cut", count))
    write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hierarchical_rows)

    covariance = (x.T @ x) / (len(x) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order_eig = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order_eig]
    eigenvectors = eigenvectors[:, order_eig]
    for component in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, component])))
        if eigenvectors[pivot, component] < 0:
            eigenvectors[:, component] *= -1
    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    projections = x @ eigenvectors[:, :2]
    results.append(("eigenvalue_sum", total))
    results.extend((f"eigenvalue_{i+1}", value) for i, value in enumerate(eigenvalues))
    results.extend((f"explained_ratio_{i+1}", value) for i, value in enumerate(ratios))
    results.extend((f"cumulative_ratio_{i+1}", value) for i, value in enumerate(cumulative))
    results.append(("components_for_80pct", int(np.flatnonzero(cumulative >= 0.8)[0] + 1)))
    results.extend((f"pc1_loading_X{i+1}", eigenvectors[i, 0]) for i in range(5))
    results.extend((f"pc2_loading_X{i+1}", eigenvectors[i, 1]) for i in range(5))
    for row in range(3):
        results.extend([(f"row{row+1}_pc1", projections[row, 0]),
                        (f"row{row+1}_pc2", projections[row, 1])])
    for ordinal, name in enumerate(segment_names):
        results.append((f"mean_pc1_segment_{name}", float(np.mean(projections[segment_ids == ordinal, 0]))))
    write_results("part3_results.txt", results)


if __name__ == "__main__":
    main()

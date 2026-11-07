"""Part 1: estimation, inference, and regression baselines."""
import numpy as np

from common import load_data, write_results


def line_fit(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    fitted = intercept + slope * x
    residual = y - fitted
    tss = np.sum((y - ybar) ** 2)
    ess = np.sum((fitted - ybar) ** 2)
    rss = np.sum(residual ** 2)
    return slope, intercept, fitted, residual, tss, ess, rss


def main():
    names, _, data = load_data()
    columns = {name: data[:, i] for i, name in enumerate(names[1:])}
    value = columns["order_value_inr_hundreds"]
    x4 = columns["avg_order_value_inr_hundreds"]
    n = value.size
    mean_value = float(np.mean(value))
    sd_value = float(np.std(value, ddof=1))
    se_value = sd_value / np.sqrt(n)
    ci_low = mean_value - 1.96 * se_value
    ci_high = mean_value + 1.96 * se_value
    z_stat = (mean_value - 8.0) / se_value

    slope, intercept, fitted, residual, tss, ess, rss = line_fit(x4, value)
    r2 = 1.0 - rss / tss
    rmse = np.sqrt(rss / n)
    mae = np.mean(np.abs(residual))

    x4_sd = np.std(x4, ddof=1)
    z = (x4 - np.mean(x4)) / x4_sd
    m = 0.0
    c = 0.0
    gd_costs = {0: np.mean(value ** 2)}
    for epoch in range(1, 501):
        error = value - (m * z + c)
        grad_m = -(2.0 / n) * np.sum(z * error)
        grad_c = -(2.0 / n) * np.sum(error)
        m -= 0.1 * grad_m
        c -= 0.1 * grad_c
        if epoch in (1, 10, 100, 500):
            gd_costs[epoch] = np.mean((value - (m * z + c)) ** 2)
    gd_slope = m / x4_sd
    gd_intercept = c - m * np.mean(x4) / x4_sd

    ordered = np.argsort(fitted, kind="stable")
    raw_lower = np.std(residual[ordered[:400]], ddof=1)
    raw_upper = np.std(residual[ordered[400:]], ddof=1)
    log_x = np.log(x4)
    log_y = np.log(value)
    ll_slope, ll_intercept, ll_fitted, ll_residual, ll_tss, _, ll_rss = line_fit(log_x, log_y)
    ll_r2 = 1.0 - ll_rss / ll_tss
    ll_rmse = np.sqrt(ll_rss / n)
    ll_ordered = np.argsort(ll_fitted, kind="stable")
    ll_lower = np.std(ll_residual[ll_ordered[:400]], ddof=1)
    ll_upper = np.std(ll_residual[ll_ordered[400:]], ddof=1)

    matrix = np.column_stack([
        np.ones(n), log_x,
        columns["customer_tenure_months"], columns["complaint_count_90d"],
        columns["discount_order_share"], columns["loyalty_score"],
        columns["cod_flag"], columns["plus_member_flag"]
    ])
    beta = np.linalg.solve(matrix.T @ matrix, matrix.T @ log_y)
    mr_fit = matrix @ beta
    mr_tss = np.sum((log_y - np.mean(log_y)) ** 2)
    mr_r2 = 1.0 - np.sum((log_y - mr_fit) ** 2) / mr_tss
    mr_adj = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
    ll_adj = 1.0 - (1.0 - ll_r2) * (n - 1) / (n - 1 - 1)

    results = [
        ("n", n), ("mean_value", mean_value), ("sd_value", sd_value),
        ("se_value", se_value), ("ci95_low", ci_low), ("ci95_high", ci_high),
        ("z_stat", z_stat), ("reject_h0", "yes" if abs(z_stat) > 1.96 else "no"),
        ("n_required", int(np.ceil((1.96 * sd_value / 0.5) ** 2))),
        ("reg_slope", slope), ("reg_intercept", intercept), ("tss", tss),
        ("ess", ess), ("rss", rss),
        ("tss_equals_ess_plus_rss", "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"),
        ("r2", r2), ("rmse", rmse), ("mae", mae),
        ("gd_J0", gd_costs[0]), ("gd_J1", gd_costs[1]),
        ("gd_J10", gd_costs[10]), ("gd_J100", gd_costs[100]),
        ("gd_J500", gd_costs[500]), ("gd_slope", gd_slope),
        ("gd_intercept", gd_intercept), ("gd_slope_gap", abs(gd_slope - slope)),
        ("resid_sd_lower_raw", raw_lower), ("resid_sd_upper_raw", raw_upper),
        ("resid_sd_ratio_raw", raw_upper / raw_lower),
        ("loglog_intercept", ll_intercept), ("loglog_slope", ll_slope),
        ("loglog_r2", ll_r2), ("loglog_rmse", ll_rmse),
        ("resid_sd_lower_log", ll_lower), ("resid_sd_upper_log", ll_upper),
        ("resid_sd_ratio_log", ll_upper / ll_lower),
    ]
    results.extend((f"mr_beta_{i}", value) for i, value in enumerate(beta))
    results.extend([("mr_r2", mr_r2), ("mr_adj_r2", mr_adj), ("loglog_adj_r2", ll_adj)])
    write_results("part1_results.txt", results)


if __name__ == "__main__":
    main()

Domain: E-commerce

This capstone uses a reproducible synthetic dataset to estimate order value, model the churn outcome, profile customer segments, and compare simple structured-data models. The records do not come from Flipkart and the results do not describe its actual customers or operations.

## How to run

Use Python 3.10 or newer. From a clean clone, run these commands in Windows PowerShell from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe part1.py
.\.venv\Scripts\python.exe part2.py
.\.venv\Scripts\python.exe part3.py
.\.venv\Scripts\python.exe part4.py
```

The generator writes `data/records.csv` and prints the row count, column count, positive-outcome count, and file hash. Each Part script writes its results to `outputs/`. Part 2 uses manual NumPy implementations of logistic regression and KNN; Part 3 implements K-means and hierarchical clustering and uses `numpy.linalg.eigh` for PCA; Part 4 reuses the Part 2 models and computes votes, boosting weights, and parameter counts. The scripts do not require a notebook or network service after dependencies are installed.

## Part 1 findings

The 800 synthetic records have a mean order value of 8.9382 hundreds of INR (about INR 894), with a 95% interval from 8.3026 to 9.5738 hundreds of INR. The test of the INR 800 planning assumption gives z = 2.8930, beyond the 1.96 critical value, so H0 is rejected at 5%. A Type I error here would mean changing the plan after rejecting INR 800 even though the true population mean is INR 800. The observed difference is 0.9382 hundreds of INR (about INR 94, or 11.7% of the assumption); it may matter at scale, but category mix and margin determine whether that difference is practically important.

The estimated sample size for a ±0.5-hundred-INR margin is 1,293, compared with 800 available rows, a shortfall of 493 for that precision target. Gradient descent and the closed-form regression agree to a slope gap of 0.0000 because repeated full-batch updates with a suitable learning rate converge to the least-squares solution.

The raw regression residual SD is 3.7525 times higher in the upper fitted-value half, showing a clear funnel. The log-log residual SD ratio is 1.0396, so logging the variables largely removes the change in spread. The multiple regression coefficients whose absolute values exceed 0.05 (excluding the constant) are ln(X4), X2 complaint count, X3 discount-order share, and X6 COD flag. Its adjusted R² is 0.7403 versus 0.6561 for the log-log model, indicating better in-sample fit; the added predictors would still need validation on new data before relying on that gain.

## Part 2 findings

After 200 updates, the losses from lowest to highest are η = 1.0 (0.3703), η = 0.1 (0.3776), and η = 0.01 (0.5091). Here the smaller rate learns more slowly, while the larger tested rate improves the loss; a still larger rate could overshoot, so this ordering does not imply that learning rate should always be increased.

The three largest absolute logistic weights are X3 discount-order share (0.9013), X1 tenure (−0.5554), and X2 complaint count (0.5505). A one-training-SD increase in discount-order share multiplies the estimated churn odds by 2.4629, with the other standardized predictors held fixed.

The all-negative baseline accuracy is 0.7875, close to the logistic model's 0.8042 at τ = 0.5; because only 51 of 240 test records are positive, accuracy can look good while missing many churn cases. Lowering τ from 0.5 to 0.3 raises recall from 0.4706 to 0.7451 and lowers precision from 0.5455 to 0.4634. At τ = 0.3, logistic regression has F1 0.5714 and recall 0.7451, compared with KNN's F1 0.3696 and recall 0.3333.

The cost-minimizing training threshold is τ = 0.15. On the test set, this logistic rule costs INR 6,050, versus INR 11,800 at τ = 0.5 and INR 20,400 if nobody is flagged. The threshold was selected on training rows to avoid tuning a decision rule to the held-out test outcomes.

## Part 3 findings

The successive WCSS drops are 659.8282 (K=2 to 3), 108.0739 (3 to 4), 171.0575 (4 to 5), and 45.5466 (5 to 6). The specified elbow rule selects K = 3 because the drop from 3 to 4 is less than half the drop from 2 to 3. The drops do not shrink steadily; the pinned starting rows make this a reproducible solution, not necessarily the global optimum.

- **A — Long-tenure, low-friction loyal shoppers:** 322 records, average tenure 47.50 months, loyalty 62.16, and outcome rate 1.86%. Offer loyalty recognition or referral benefits and monitor service quality.
- **B — High-value, high-friction shoppers:** 255 records, complaint count 6.07 and mean order value INR 1,712. Route unresolved complaints to service recovery before offering a retention incentive.
- **C — Discount-led, newer at-risk shoppers:** 223 records, discount-order share 0.7118 and outcome rate 46.64%, the highest of the three segments. Test a targeted retention offer and measure whether it prevents churn at an acceptable cost.

At the largest-gap cut, single linkage gives {E1}, {E2,E3,E6}, {E4}, and {E5}; complete linkage gives {E1,E4,E5} and {E2,E3,E6}. Single linkage joins groups when any pair is close, so the E2–E3–E6 chain forms early. Complete linkage considers the farthest cross-cluster pair, keeping groups separate until the large final merge height.

The first PCA component contrasts higher tenure and loyalty (positive loadings 0.5317 and 0.4418) with higher complaints, discount share, and average order value (negative loadings −0.4100, −0.5187, and −0.2916). Three components retain at least 80% of variance (86.16%). Mean PC1 scores are 1.7773 for A, −0.7831 for B, and −1.6709 for C, separating the long-tenure group from the shorter-tenure, more discount-oriented groups along this axis. Standardization makes the feature mean vector zero before PCA.

## Part 4 findings

S4 (X1 ≤ 24) is the best candidate split by both information gain (0.1252) and Gini gain (0.0606), the largest gains among the four tested rules. A tree grown until every leaf is pure can memorize noise and small-sample quirks, which harms performance on new records. A maximum-depth limit or a minimum leaf size can stop it growing too deeply.

At the required τ = 0.3 comparison point, logistic regression is the strongest F1 model: F1 0.5714 and recall 0.7451. Combining models did not improve it: hard voting reaches F1 0.5481 and recall 0.7255, while soft voting reaches F1 0.5507 and recall 0.7451.

After round 1, the two points misclassified by h1 carry a combined normalized weight of 0.5000. Round 2 assigns h2 a larger coefficient (α₂ = 0.8047 versus α₁ = 0.5493) because h2 has the lower weighted error (0.1667 versus 0.2500).

For the image-sized Model A convolutional option, the output shapes and parameter counts are:

| Layer | Output shape | Parameters |
|---|---:|---:|
| Conv1 (5×5, 6 filters, padding 2) | 28×28×6 | 156 |
| ReLU | 28×28×6 | 0 |
| Max-pool 1 (2×2, stride 2) | 14×14×6 | 0 |
| Conv2 (5×5, 16 filters, no padding) | 10×10×16 | 2,416 |
| ReLU | 10×10×16 | 0 |
| Max-pool 2 (2×2, stride 2) | 5×5×16 | 0 |
| Flatten | 1×1×400 | 0 |
| Dense1 + ReLU | 1×1×32 | 12,832 |
| Dense output | 1×1×10 | 330 |

For Model B, W_xh has shape 128×64, W_hh is 128×128, and W_hy is 16×128. The RNN has 26,768 parameters; the same matrices are reused at every sequence step, so parameter count does not grow with sequence length. For ten classes, use ReLU in hidden layers and softmax with categorical cross-entropy at the output. In a very deep plain network, repeated small derivatives can shrink the gradient before it reaches early layers. A skip connection adds a direct path for information and gradients, helping those layers keep learning.

## Model-choice memo

**To: Head of the Retention Team**

I recommend logistic regression for the structured-data pilot, using the cost-selected threshold τ = 0.15. At τ = 0.3 it has the highest F1 among the five comparisons (0.5714), with recall 0.7451; lowering the threshold to 0.15 reduces the held-out cost to INR 6,050, compared with INR 11,800 at 0.5 and INR 20,400 when nobody is flagged. The model's cost threshold was selected using training records, and the held-out figures are a single split rather than a stable estimate of future costs. These data are synthetic, so validate on representative real historical records before sending offers. Neither Model A (product images) nor Model B (review topics) should be built ahead of validating the structured retention model; revisit those use cases after that validation establishes a reliable baseline.

## Limitations

- The generator creates synthetic records; the segments, coefficients, churn rates, and costs do not represent Flipkart's real customers or policies.
- Classification uses one pinned 70/30 split and only 51 positive test outcomes, so reported metrics and costs can vary on another sample.
- The money rule simplifies coupon and missed-customer costs to fixed INR 50 and INR 400 amounts; it omits offer uptake, customer lifetime value, and operational costs.
- K-means uses specified starting rows and is reproducible but may not find the global minimum WCSS; the segment names are interpretations of averages, not ground-truth customer types.
- Regression and model comparisons do not establish causal effects, and the richer Part 1 regression fit has not been tested on a separate validation dataset.

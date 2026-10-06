# Interview Defense Guide

This document prepares you for technical grilling on every design choice in the project.

## 1. Feature Engineering
**Q: How did you ensure no future leakage in your rolling window features?**
*A: I used DuckDB for SQL-based feature engineering and strictly defined my windows as `ROWS BETWEEN X PRECEDING AND CURRENT ROW`. I wrote a regex-based pytest (`test_leakage.py`) that fails the build if the word `FOLLOWING` ever appears in a window definition. This guarantees the model never peeks at future statements.*

**Q: Why compute features in SQL (DuckDB) rather than Pandas?**
*A: The raw dataset was 47GB of CSVs, which exceeds the RAM on standard laptops. Pandas would throw an Out-Of-Memory (OOM) error. DuckDB allowed me to stream the data out-of-core directly into compressed Parquet files, executing complex window aggregations with minimal memory footprint.*

**Q: A customer has only 3 statements. What happens to your 6-statement rolling mean?**
*A: SQL window functions handle this gracefully. `BETWEEN 5 PRECEDING AND CURRENT ROW` will simply take the average of the 3 available rows. It doesn't output NULL, it just calculates the mean over the available history up to that point.*

## 2. Modeling & Metrics
**Q: Why did you apply a 20x weight to the negative class?**
*A: The Amex competition dataset artificially subsampled the negative class (good accounts) by a factor of 20 (5% retention). If I trained or evaluated without applying a 20x weight to negatives, the model's predicted probabilities would be massively miscalibrated, overestimating default risk. This ruins downstream profit optimization.*

**Q: Did you run a statistical test on the AUC difference between LR and LightGBM?**
*A: Yes, I implemented a fast DeLong test. However, it highlighted a major flaw in relying on statistical tests for real-world imbalanced data: standard DeLong implementations compute variance based on unweighted rank-sums. Applying it to our 20x subsampled dataset computes variance on the wrong population, yielding an invalid p-value. This proved exactly why we must rely on Expected Profit curves rather than pure statistical metrics.*

## 3. Decisioning & Economics
**Q: What is the exact formula for expected profit per account?**
*A: Expected Profit = $(1 - p) \times M - p \times L$, where $p$ is the probability of default, $M$ is the margin for a good account, and $L$ is the loss given default. Setting this to >0 gives the optimal cutoff: $p < \frac{M}{L + M}$. Dividing by M gives $p < \frac{1}{r + 1}$, where $r = L/M$.*

**Q: Why is F1-score a terrible cutoff for credit risk?**
*A: F1 implicitly assumes false positives and false negatives are equally bad (or tied to the class balance). In our stress scenario, a default costs 45 times more than the margin gained from a good account. F1 picked a threshold of 25.8%, which approved far too many risky accounts, resulting in a 31.8% loss in potential profit compared to the optimal threshold.*

**Q: Explain MiniMax Regret to a non-technical stakeholder.**
*A: "We don't know exactly what the economy will do next year, so we can't perfectly optimize our approval cutoff. Instead of guessing, we found a 'safe' cutoff. The MiniMax Regret cutoff guarantees that whether the economy booms or crashes, we will never lose more than 4.7% of our maximum possible profit. It bounds our worst-case scenario."*

## 4. Calibration & Statistical Rigor
**Q: Why calibrate the model if your AUC was already 0.958?**
*A: AUC only measures how well the model ranks customers, but our Expected Profit equation requires absolute probabilities to determine if a customer crosses the profit threshold. An uncalibrated model with a perfect AUC could still lose millions. We used Isotonic Regression to drop the Expected Calibration Error (ECE) to near zero.*

**Q: Why use a Bootstrap Confidence Interval for profit instead of a simple t-test?**
*A: Credit profit distributions are incredibly heavy-tailed (most customers pay a small margin, a few default with a massive loss). A standard t-test assumes a normal distribution and would severely underestimate the variance. Bootstrapping makes no distributional assumptions, making it the only safe choice for credit economics.*

## 5. Monitoring & Explainability
**Q: How do you know when to retrain the model in production?**
*A: We monitor feature drift using the Population Stability Index (PSI). We break distributions into deciles and compare the training set to recent applications. If PSI > 0.25 on core features (like Delinquency `D_` variables), it triggers a red flag for retraining.*

**Q: How do you provide reason codes if your features are anonymized?**
*A: Using SHAP's TreeExplainer, we extracted the raw feature importances for each individual customer. Even though we don't know exactly what `D_39` is, we know `D` stands for Delinquency. We aggregated SHAP values by their prefix families to provide explainable categories like "Recent Delinquency Indicators" or "Payment-to-Balance Metrics".*

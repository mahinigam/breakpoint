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

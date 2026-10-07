# Interview Defense Guide

This document prepares you for technical grilling on every design choice in the project. It contains 5 questions per phase across all 10 phases.

## Phase 0: Project Setup & Architecture
**Q1: Why did you use a Makefile for a Python project?**
*A: It standardizes the execution pipeline. Instead of relying on a complex bash script or asking users to run Python scripts in a specific order, `make all` guarantees the DAG (Data -> Features -> Train -> Evaluate) executes correctly.*

**Q2: How do you manage configurations across different scripts?**
*A: I use YAML files (`settings.yaml` and `economic_scenarios.yaml`) loaded via a central `src/config.py`. This ensures magic numbers (like the 20x negative weight) are defined exactly once, preventing silent mathematical errors.*

**Q3: Why not just use Jupyter Notebooks?**
*A: Notebooks are great for EDA but terrible for production pipelines. They suffer from hidden state and out-of-order execution. I structured this as a modular Python package (`src/`) so it can be unit-tested, version-controlled cleanly, and run via CI/CD.*

**Q4: How did you ensure your environment is reproducible?**
*A: I provided a strict `requirements.txt` with exact versions (e.g., LightGBM, DuckDB, SHAP) and automated the setup process using `make setup` in a virtual environment. I also seed all random operations.*

**Q5: What is the purpose of the `.github/workflows/ci.yml`?**
*A: It enforces Continuous Integration. On every push to main, it provisions an Ubuntu runner, installs dependencies, and runs the PyTest suite. This ensures I never push broken code or data leakage regressions.*

## Phase 1: Data Engineering & SQL Features
**Q1: How did you ensure no future leakage in your rolling window features?**
*A: I strictly defined my SQL windows as `ROWS BETWEEN X PRECEDING AND CURRENT ROW`. I also wrote a regex-based pytest (`test_leakage.py`) that fails the build if the word `FOLLOWING` ever appears in a window definition.*

**Q2: Why compute features in SQL (DuckDB) rather than Pandas?**
*A: The raw dataset was 47GB of CSVs. Pandas would throw an Out-Of-Memory (OOM) error on most laptops. DuckDB allowed me to stream the data out-of-core directly into compressed Parquet files, executing complex aggregations with minimal RAM.*

**Q3: A customer has only 3 statements. What happens to your 6-statement rolling mean?**
*A: SQL window functions handle this gracefully. `BETWEEN 5 PRECEDING AND CURRENT ROW` will take the average of the 3 available rows. It doesn't output NULL, it calculates the mean over the available history up to that point.*

**Q4: Why did you downcast float64 to float32?**
*A: Credit bureau features rarely need double-precision floats. Downcasting to float32 cut the memory footprint of the feature matrix in half with zero loss in predictive performance.*

**Q5: How do you handle the chronological ordering of customer statements?**
*A: Before applying any window functions, the DuckDB query partitions by `customer_ID` and strictly orders by `S_2` (the statement date). This ensures historical features are mathematically sound.*

## Phase 2: Modeling & Metrics
**Q1: Why did you apply a 20x weight to the negative class?**
*A: The Amex dataset artificially subsampled the negative class (good accounts) by a factor of 20 (5% retention). If I trained without applying this weight, the predicted probabilities would be miscalibrated, ruining downstream profit optimization.*

**Q2: Why use LightGBM instead of XGBoost or a Neural Network?**
*A: LightGBM natively handles NaN values (very common in credit data) and categorical features without requiring massive one-hot encoding matrices. It builds trees histogram-wise, making it exceptionally fast for 400+ features and 450k rows.*

**Q3: Did you run a statistical test on the AUC difference between LR and LightGBM?**
*A: Yes, I implemented a fast DeLong test. However, it highlighted a major flaw: standard DeLong implementations compute variance based on unweighted rank-sums. Applying it to our 20x subsampled dataset yields an invalid p-value. This proved why we must rely on Expected Profit.*

**Q4: Why include a Logistic Regression baseline if you knew LightGBM would win?**
*A: To quantify the "complexity premium." By comparing LightGBM to LR, I can prove to stakeholders exactly how much extra profit the non-linear model generates, justifying the added complexity in deployment.*

**Q5: How did you evaluate the model to prevent overfitting?**
*A: I used 5-Fold Stratified Cross Validation. This ensures the 20x class imbalance is preserved across folds, and out-of-fold (OOF) predictions are strictly separated from training data.*

## Phase 3: Calibration
**Q1: Why calibrate the model if your AUC was already 0.958?**
*A: AUC only measures ranking (who is more risky). But Expected Profit requires absolute probabilities (is the risk exactly 4.2%?). An uncalibrated model with perfect AUC can still lose millions if it systematically over- or under-predicts default probabilities.*

**Q2: What is Expected Calibration Error (ECE)?**
*A: ECE bins predictions into deciles and measures the average absolute difference between the predicted probability and the actual default rate in that bin. We want it as close to 0 as possible.*

**Q3: Why did Isotonic Regression beat Platt Scaling?**
*A: Platt Scaling assumes the model's scores can be mapped to probabilities using a sigmoid function (logistic regression). LightGBM trees often produce non-sigmoid score distributions. Isotonic Regression is non-parametric (a step function) and perfectly fits complex tree outputs.*

**Q4: Can calibration hurt AUC?**
*A: Both Isotonic and Platt scaling are monotonic transformations. They preserve the exact rank ordering of the predictions, so AUC remains mathematically identical. They only adjust the absolute scale.*

**Q5: How do you calibrate out-of-fold predictions without leakage?**
*A: You must fit the calibrator on the hold-out validation fold, not the training fold. My pipeline calibrates the validation fold using a model trained on the training fold, ensuring strict separation.*

## Phase 4: Economics & Decisioning
**Q1: What is the exact formula for expected profit per account?**
*A: Expected Profit = $(1 - p) \times M - p \times L$, where $p$ is the probability of default, $M$ is the margin, and $L$ is the loss given default. The optimal cutoff is $p < \frac{1}{r + 1}$, where $r = L/M$.*

**Q2: Why is F1-score a terrible cutoff for credit risk?**
*A: F1 assumes false positives and false negatives have equal costs (or costs tied to class balance). In our stress scenario, a default costs 45 times more than a good account margin. F1 ignored this, resulting in a 31.8% profit loss.*

**Q3: Explain MiniMax Regret to a non-technical stakeholder.**
*A: "We don't know exactly what the economy will do next year. Instead of guessing, we found a 'safe' cutoff. The MiniMax Regret policy guarantees that whether the economy booms or crashes, we will never lose more than 4.7% of our maximum possible profit."*

**Q4: How did you calculate fractional regret?**
*A: Fractional Regret = `(Maximum Possible Profit - Actual Profit) / Maximum Possible Profit`. We normalize it so a high-dollar Aggressive scenario doesn't completely overshadow a low-dollar Stress scenario.*

**Q5: What does the Sensitivity Sweep prove?**
*A: It proves that as the economy shifts (cost ratio `r` moves from 1 to 50), statistical heuristics randomly collapse, while the MiniMax policy stays consistently near 100% of the theoretical maximum profit.*

## Phase 5: Statistical Rigor
**Q1: Why use a Bootstrap Confidence Interval for profit instead of a simple t-test?**
*A: Credit profit distributions are incredibly heavy-tailed. A standard t-test assumes a normal distribution and underestimates variance. Bootstrapping makes no distributional assumptions.*

**Q2: How did you implement the Paired Bootstrap?**
*A: I resampled the *customers* (with replacement) 1000 times. For each sample, I computed the profit of the MiniMax policy minus the profit of the F1 policy. This yields a distribution of the *difference* in profit.*

**Q3: What is Champion-Challenger Test Design?**
*A: It's an A/B test for credit models. The Champion is the existing policy, the Challenger is the new MiniMax policy. You randomly assign accounts to each to measure true causal lift.*

**Q4: Why simulate Power Analysis instead of using a formula?**
*A: Standard power formulas (like Cohen's d) assume normality. By simulating the A/B test using actual bootstrap samples of our heavy-tailed predictions, we get a much more accurate estimate of required sample size.*

**Q5: What is the risk of deploying without an A/B test?**
*A: "Winner's Curse" and selection bias. Our offline expected profit assumes the future population looks exactly like the past. A live champion-challenger test proves the model actually generates real-world cash.*

## Phase 6: Monitoring & Drift
**Q1: What is the Population Stability Index (PSI)?**
*A: PSI measures how much a feature's distribution has shifted between training and production. It bins the data and computes `sum((Actual% - Expected%) * ln(Actual% / Expected%))`.*

**Q2: What is your threshold for triggering a retrain?**
*A: A standard industry threshold is PSI > 0.25 for severe shift. I trigger a review if >10% of features drift, or if any 3 features in the critical Delinquency/Balance families drift.*

**Q3: Why calculate PSI on a 1% sample?**
*A: Due to disk space and memory constraints on a local machine, processing the full test set repeatedly was unfeasible. In production, we would compute this on a 100% daily or weekly snapshot.*

**Q4: Can a model degrade even if PSI is 0?**
*A: Yes, if the relationship between the features and the target changes (concept drift). PSI only measures feature drift (covariate shift).*

**Q5: How would you monitor concept drift in credit?**
*A: Because defaults take 6-12 months to mature, we cannot monitor concept drift immediately. We would monitor early warning indicators (e.g., 30-days past due rates at 3 months on book).*

## Phase 7: Explainability
**Q1: How do you provide reason codes if your features are anonymized?**
*A: Using SHAP, we extract raw feature importances. Even though we don't know exactly what `D_39` is, we know `D` stands for Delinquency. We aggregated SHAP values by prefix families to provide categories like "Recent Delinquency Indicators".*

**Q2: Why use SHAP TreeExplainer instead of exact SHAP or LIME?**
*A: LIME is a linear approximation and can be unstable. Exact SHAP is exponentially slow. TreeExplainer uses the internal structure of LightGBM trees to compute exact SHAP values in polynomial time.*

**Q3: How do you handle negative SHAP values for reason codes?**
*A: Reason codes are for explaining why an account was rejected or restricted. Therefore, we only look at features with positive SHAP values (features that pushed the predicted default risk *higher*).*

**Q4: What is the difference between Global and Local explainability?**
*A: Global explainability (Feature Importance plots) tells us how the model works on average. Local explainability (Reason Codes for one applicant) tells us why a specific decision was made.*

**Q5: Can SHAP prove causality?**
*A: No. SHAP attributes the model's prediction to input features based on correlation. If 'high balance' correlates with default, SHAP highlights it, but it doesn't prove that lowering the balance prevents default.*

## Phase 8: Dashboards & Reporting
**Q1: Why did you build an export script instead of connecting Tableau directly to DuckDB?**
*A: BI tools struggle with 400-column, 450k-row raw datasets. The export script pre-aggregates the profit curves, regret maps, and PSI metrics into tiny, fast CSVs so the executive dashboard loads instantly.*

**Q2: What are the 4 main panels of your Executive Dashboard?**
*A: 1. Profit Curves under 4 economic scenarios. 2. Regret Heatmap showing robustness. 3. Calibration reliability plot. 4. Drift Monitoring summary.*

**Q3: Who is the audience for this dashboard?**
*A: Chief Risk Officers and Portfolio Managers. They don't care about AUC; they care about downside protection, margin, and when the model will break.*

**Q4: Why not use a Python dashboarding tool like Streamlit?**
*A: Streamlit is great for technical prototypes, but Tableau/PowerBI are the enterprise standard in finance for executive reporting.*

**Q5: How do you prevent stakeholders from misinterpreting the Profit numbers?**
*A: The dashboard explicitly labels all numbers as "Simulated Expected Profit" based on specified margin assumptions, not actual historical P&L.*

## Phase 9: Engineering Hygiene & Tests
**Q1: What does `test_determinism.py` do?**
*A: It verifies that running the pipeline twice yields the exact same model and predictions. It catches unseeded random number generators or non-deterministic SQL aggregations.*

**Q2: Why write unit tests for economics?**
*A: The expected profit math is the core thesis of the project. `test_economics.py` asserts that profit is strictly monotonically decreasing as Loss Given Default (LGD) increases, catching basic arithmetic bugs.*

**Q3: What is the value of `pytest` fixtures?**
*A: In `conftest.py`, I load the 1% sample data once into memory and pass it to multiple tests. This prevents tests from taking 5 minutes to run due to repeated disk I/O.*

**Q4: How does your CI/CD pipeline handle secrets?**
*A: This specific public repo doesn't require secrets, but in production, API keys (e.g., for database access or MLflow tracking) are stored in GitHub Secrets and injected into the runner environment.*

**Q5: If I clone this repo, what is the exact command to replicate your results?**
*A: `make all`. It sets up the environment, converts the data, builds features, trains the models, and generates every plot and markdown report automatically.*

## Phase 10: Deliverables & Communication
**Q1: Why write a Decision Memo?**
*A: Data Scientists often fail because they only communicate in technical metrics. A 1-page Decision Memo translates an AUC of 0.958 into a clear business recommendation: deploy the 0.0500 threshold to protect against recession.*

**Q2: What is a Model Card?**
*A: Introduced by Google, a Model Card is a standardized document detailing the model's intended use, training data, ethical considerations, and known limitations. It ensures transparency for compliance teams.*

**Q3: What is the most critical limitation of this project?**
*A: The lack of real dollar values. We used assumed economic scenarios. While the MiniMax methodology is robust, the actual 0.0500 cutoff cannot be blindly deployed without plugging in Amex's true margins and LGDs.*

**Q4: How did you ensure every number in your resume bullets is verifiable?**
*A: The `RESULTS.md` file acts as the source of truth. Every metric cited in the resume (e.g., 31.8% profit loss, 99.8% power with 10k accounts) is generated by the pipeline and logged there.*

**Q5: How would this project change if you built it for a real bank today?**
*A: I would use a non-anonymized dataset to build a proper Weight of Evidence (WoE) scorecard for regulatory compliance, integrate macroeconomic variables (like interest rates), and deploy via a real-time feature store.*

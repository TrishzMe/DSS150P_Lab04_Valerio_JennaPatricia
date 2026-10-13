# Predicting PM2.5 Air Quality Exceedances Using Machine Learning: A Co-Pollutant Chemistry Approach

**Seancarlo Raphael S. Grino, Bienn Jaime P. Perez, Jenna Patricia G. Valerio**
School of Data Science, Mapúa University, Makati, Philippines

---

## Abstract

Fine particulate matter (PM2.5) is a leading air-quality health hazard, but dedicated particulate sensors remain expensive and unevenly deployed relative to cheaper, widely available gas sensors (NO2, O3, SO2, CO). This study was originally proposed to predict PM2.5-related air quality from climate variables (temperature, humidity, wind, precipitation), following the dominant approach in prior literature. However, profiling of the World Air Quality dataset (OpenAQ network aggregation, 54,255 raw readings, 40+ countries) revealed only 117 temperature and 66 relative-humidity readings globally — far too sparse to support climate-driven modeling. We therefore redirected the study toward a co-pollutant chemistry approach: predicting whether a station-day's PM2.5 concentration exceeds the World Health Organization's 24-hour guideline (15 µg/m³) using only co-monitored gaseous pollutants. After cleaning and pivoting, 2,628 model-ready station-days were retained, with two engineered atmospheric-chemistry features — total oxidant capacity (NO2 + O3) and an NO2/O3 ratio — added to the raw gas concentrations. Three classifiers were trained and compared: Logistic Regression, Random Forest, and Gradient Boosting, validated via a stratified 80/20 split and 5-fold stratified cross-validation. Random Forest achieved the highest held-out test accuracy (71.7%) and Gradient Boosting the highest cross-validated F1 (0.712), both outperforming the Logistic Regression baseline (67.3%) by 4–5 points. Feature importance analysis showed ozone-related terms (O3, total oxidant, NO2/O3 ratio) dominate the prediction, ahead of any single raw pollutant, consistent with secondary particulate formation via ozone–NOx photochemistry. These results indicate that co-monitored gas chemistry is a viable, low-cost proxy for PM2.5 exceedance risk where dedicated particulate sensors are unavailable.

**Keywords:** PM2.5, air quality prediction, machine learning, random forest, gradient boosting, atmospheric chemistry, feature engineering

---

## I. Introduction

Fine particulate matter under 2.5 micrometers in diameter (PM2.5) penetrates deep into the human respiratory and circulatory systems and is one of the most consistently cited drivers of air-quality-related morbidity and mortality worldwide. The World Health Organization (WHO) sets a 24-hour exposure guideline of 15 µg/m³, above which health risk is considered elevated. Despite its importance, dedicated PM2.5 monitoring hardware is costly and unevenly deployed globally, whereas sensors for co-monitored gaseous pollutants — nitrogen dioxide (NO2), ozone (O3), sulfur dioxide (SO2), and carbon monoxide (CO) — are considerably more common at regulatory air-quality stations.

This project was originally proposed under the title *"Predicting Air Pollution Impacts from Climate Variables,"* following the dominant paradigm in the literature: using meteorological variables (temperature, relative humidity, wind speed, precipitation) as predictors of PM2.5 and related air-quality indices. During data preparation, however, profiling of the selected dataset revealed that temperature and relative humidity were reported only 117 and 66 times, respectively, across the entire 54,255-row dataset — effectively single, unrepeated sensor snapshots contributed by a handful of IoT devices, and far too sparse to support any reliable model. Rather than force a model onto data that could not support it, this study redirects the independent-variable side of the investigation toward the predictors that were in fact well populated: co-monitored gaseous pollutants and chemistry-informed features derived from them.

**The research question addressed in this paper is:**

> Can co-monitored gaseous pollutants (NO2, O3, SO2, CO) reliably predict whether a location's PM2.5 concentration exceeds the WHO 24-hour health guideline (15 µg/m³), allowing monitoring stations without a dedicated particulate sensor to flag likely air-quality health risks using sensors they already have?

**The contributions of this paper are as follows:**

1. An empirical demonstration that the climate-variable-driven approach dominant in prior PM2.5 literature is infeasible on a real-world, single-snapshot, multi-country air-quality aggregation, motivating a substitute predictor set.
2. Two chemistry-informed engineered features — total oxidant capacity and an NO2/O3 traffic-versus-regional-ozone ratio — constructed from raw co-pollutant concentrations rather than using raw concentrations alone.
3. A comparative evaluation of a linear baseline (Logistic Regression) against two tree-based ensembles (Random Forest, Gradient Boosting) for WHO-threshold PM2.5 exceedance classification, validated with both a held-out test split and 5-fold stratified cross-validation.
4. Model interpretation showing that ozone-related chemistry, rather than any single raw pollutant, is the dominant predictive signal.

---

## II. Related Work

Fifteen studies were reviewed to establish the state of the art in PM2.5 and air-quality prediction. The majority of these studies use meteorological variables as the primary predictors of PM2.5. Ramos and Varquez [1] used XGBoost with Optuna hyperparameter optimization on satellite and climate reanalysis data (temperature, pressure, wind speed, precipitation) to predict PM2.5 in Metro Manila, achieving R² = 0.73 and finding that precipitation reduces PM2.5 via atmospheric washout while elevated temperature increases accumulation. Caido et al. [2] compared Random Forest, SVM, and XGBoost using satellite aerosol optical depth and meteorological variables, finding wind speed the most important predictor via SHAP analysis. Salvador et al. [3] examined long-term PM and ozone trends in a tropical city, linking dry-season conditions to higher particulate retention. Bagtasa et al. [4] used wind back-trajectory analysis to show monsoon-driven seasonal variation in PM2.5 in the northwestern Philippines. Viñas et al. [5] forecast PM2.5/PM10 and AQI in Manila using an artificial neural network trained on hourly meteorological time series. Bui et al. [6] proposed STAR, a multimodal CNN-RNN architecture combining meteorology, pollutant concentration, and spatial dependencies. Megaritis et al. [7] modeled the effect of European weather patterns on PM2.5 using atmospheric chemistry transport models. Orru et al. [8], Guo et al. [9], Tai et al. [10], and Morshed et al. [11] each further reinforce that temperature, humidity, wind, and precipitation are treated as the primary drivers of particulate and ozone behavior across different regions and timescales.

A smaller subset of the reviewed literature is directly relevant to methodology independent of the climate-variable question. Berrocal et al. [12] compared statistical methods (generalized additive models, kriging) against machine learning approaches (random forests, neural networks) for national PM2.5 mapping, finding machine learning generally superior at capturing nonlinear and spatial relationships while noting the interpretability advantage of statistical models — directly motivating this study's comparison of a linear baseline against tree-based ensembles. Tırınk [13] compared XGBoost, LightGBM, and SVM for air quality index forecasting, finding gradient boosting methods consistently superior, which parallels this study's finding that tree-based ensembles outperform a linear baseline. Lu, Zhang, and Shen [14] reviewed the role of meteorology in tropospheric ozone formation, establishing that ozone chemistry is highly sensitive to temperature, solar radiation, and atmospheric circulation — providing the physical basis for this study's ozone-derived engineered features even though this study does not use meteorological variables directly. An anonymous 2023 preprint [15] proposed evaluating PM2.5 forecasts by their decision-relevance rather than purely statistical accuracy, motivating this study's framing of PM2.5 prediction as a health-guideline exceedance classification problem rather than a raw regression task.

**Positioning relative to prior work.** Using co-monitored pollutants to predict PM2.5 is not, by itself, a novel technique; several published works predict PM2.5 from raw SO2, NO2, CO, and O3 concentrations, typically alongside meteorological variables, using random forests or gradient boosting. This study's contribution is narrower and specific: (1) engineered chemistry features (total oxidant capacity, NO2/O3 ratio) rather than raw concentrations alone, and (2) an empirical demonstration, on a real-world global aggregation dataset, that the climate-variable approach dominant in the reviewed literature is infeasible when meteorological instrumentation is sparse, together with validation that a chemistry-driven substitute is viable in that scenario.

---

## III. Methodology

### A. Dataset

The World Air Quality dataset (OpenAQ network aggregation, sourced via Kaggle [16]) contains 54,255 raw pollutant readings spanning more than 40 countries, with ten raw fields: country code, city, location, coordinates, pollutant, source name, unit, value, timestamp, and country label.

### B. Data Cleaning

Readings were filtered to six pollutants of interest (PM2.5, PM10, NO2, O3, SO2, CO). Timestamps were parsed to UTC and duplicate sensor readings (same location, pollutant, and timestamp) were removed. To avoid mixing unit systems, only readings reported in µg/m³ were retained (mixed-unit ppm rows were dropped). Negative sensor readings were removed as invalid. Missing geographic labels were imputed with a placeholder value prior to pivoting, since pivot operations silently drop rows with missing index values. This yielded 30,545 cleaned readings from the original 54,255.

Readings were then aggregated to the Location + Date level (rather than exact timestamp) and pivoted into a wide format with one column per pollutant, yielding 11,558 station-days. A verification check confirmed that the median station reports on only a single date, indicating this dataset is effectively a single global snapshot per station rather than a repeated time series; accordingly, no lag or seasonal features were engineered.

### C. Feature Engineering

Rows were required to have PM2.5, NO2, and O3 jointly present, yielding 2,628 model-ready station-days. SO2 and CO were available for a smaller subset; rather than discarding rows lacking them, missing values were median-imputed and a binary missingness indicator (`SO2_missing`, `CO_missing`) was added for each, preserving the larger NO2/O3 population while allowing the model to use SO2/CO signal where genuinely available.

Two engineered features were constructed to capture atmospheric-chemistry relationships not evident in any single raw concentration:

- `Ox_total_oxidant` = NO2 + O3, the standard atmospheric total-oxidant indicator.
- `NO2_O3_ratio` = NO2 / (O3 + 1), a proxy distinguishing local traffic influence (elevated NO2) from regional photochemical ozone.

PM10 was deliberately excluded from the feature set. Exploratory analysis (Figure 1) shows PM2.5 and PM10 correlate at r = 0.72 within this dataset; combined with the fact that PM10 physically subsumes the PM2.5 size fraction by definition (particulate matter under 10 µm includes particulate matter under 2.5 µm), including PM10 as a predictor would risk leaking information about the target rather than testing an independent relationship.

**[Insert Figure 1 here: correlation_heatmap.png — Correlation heatmap of PM2.5 against co-monitored pollutants]**

The final eight-feature set is: NO2, O3, SO2, CO, `SO2_missing`, `CO_missing`, `Ox_total_oxidant`, `NO2_O3_ratio`.

### D. Target Definition

The continuous PM2.5 concentration was converted to a binary label, `Air_Quality`, using the WHO 24-hour guideline of 15 µg/m³: *Safe* (≤ 15 µg/m³) versus *Unhealthy* (> 15 µg/m³). The resulting class distribution is moderately imbalanced: 1,115 Safe (42.4%) and 1,513 Unhealthy (57.6%) station-days.

### E. Model Development

Three classifiers were trained: Logistic Regression (linear baseline, `max_iter=1000`), Random Forest (200 trees), and Gradient Boosting (200 stages). Data were split 80/20 into training (2,102 rows) and test (526 rows) sets using stratified, shuffled sampling (`random_state=42`) to preserve class balance in both sets.

### F. Evaluation Strategy

Models were evaluated on the held-out test set using accuracy, weighted precision, weighted recall, and weighted F1-score. To guard against noise from a single 20% split, 5-fold stratified cross-validation (F1-weighted scoring) was additionally performed across the full 2,628-row dataset. Confusion matrices and Random Forest feature importances were used to interpret model behavior.

---

## IV. Results

### A. Held-Out Test Performance

**Table 1. Held-Out Test Set Performance (n = 526)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.673 | 0.674 | 0.673 | 0.657 |
| **Random Forest** | **0.717** | **0.716** | **0.717** | **0.716** |
| Gradient Boosting | 0.705 | 0.704 | 0.705 | 0.704 |

Both tree-based models outperform the Logistic Regression baseline by 4–5 percentage points. Random Forest achieves the highest test accuracy (0.717), narrowly ahead of Gradient Boosting (0.705).

**[Insert Figure 2 here: model_accuracy_comparison.png — Held-out test accuracy by model]**

### B. Cross-Validation

**Table 2. 5-Fold Stratified Cross-Validation (F1-Weighted)**

| Model | CV F1 Mean | CV F1 Std |
|---|---|---|
| Logistic Regression | 0.673 | 0.023 |
| Random Forest | 0.708 | 0.008 |
| **Gradient Boosting** | **0.712** | 0.006 |

Gradient Boosting achieves the highest mean CV F1 (0.712), with Random Forest close behind (0.708); both substantially exceed Logistic Regression (0.673). The standard deviations for the two tree-based models (0.006–0.008) are an order of magnitude smaller than their ~0.5-point difference from each other, indicating that the choice between Random Forest and Gradient Boosting is not statistically meaningful here — the substantive finding is linear versus non-linear, not one ensemble method versus the other.

**[Insert Figure 3 here: confusion_matrices.png — Confusion matrices for all three classifiers]**

The confusion matrices show that Logistic Regression under-predicts the Safe class relative to the tree-based models (98 true Safe predicted correctly versus 146 for Random Forest and 141 for Gradient Boosting, out of 223 true-Safe test cases), indicating the linear boundary is comparatively biased toward the majority Unhealthy class.

### C. Feature Importance

**[Insert Figure 4 here: random_forest_key_features.png — Random Forest feature importances]**

Random Forest feature importances show ozone-related terms dominate: O3 (0.183), `Ox_total_oxidant` (0.180), and `NO2_O3_ratio` (0.180) rank highest, ahead of CO (0.157), NO2 (0.157), and SO2 (0.121). The two missingness indicators contribute least (`CO_missing` = 0.012, `SO2_missing` = 0.010), consistent with their role as imputation flags rather than physical measurements.

---

## V. Discussion

The dominance of ozone-related features — rather than any single raw gas — indicates that PM2.5 exceedance in this dataset is best explained by the broader ozone–NOx photochemical and combustion mixture at a station, consistent with how secondary particulate matter forms in the atmosphere through gas-phase chemical reactions rather than through any single precursor acting in isolation. This is consistent with the physical mechanisms described in the ozone-chemistry literature reviewed in Section II [14], even though that literature frames ozone formation as climate-driven rather than as a PM2.5 predictor per se.

**Practical implication.** A monitoring station that already reports NO2 and O3, but lacks a dedicated PM2.5 sensor, could apply a model of this kind to flag likely PM2.5 exceedances without new hardware, extending effective air-quality-health coverage using instrumentation that is already comparatively common and inexpensive.

**Limitations.** This dataset is a single global snapshot rather than a repeated-measures time series (median one reporting date per station), so no genuine temporal or lag features could be engineered, and the model cannot capture seasonal or diurnal pollution cycles. Readings are pooled across 40+ countries and monitoring-network standards, introducing measurement heterogeneity that likely caps achievable accuracy. SO2 and CO were imputed for a meaningful share of rows; performance on stations with genuine SO2/CO instrumentation may differ from the imputed cases. Held-out accuracy in the 67–72% range reflects a genuinely difficult, noisy, real-world classification problem rather than a controlled benchmark, and should be read as a proof of concept rather than an operational-grade system.

---

## VI. Conclusion and Future Work

This study set out to predict PM2.5-related air-quality risk from climate variables, following the dominant approach in the literature, but was redirected after data profiling showed the available meteorological fields (117 temperature and 66 humidity readings, globally) were too sparse to model. The resulting chemistry-driven approach — predicting WHO-threshold PM2.5 exceedance from co-monitored gaseous pollutants and two engineered atmospheric-chemistry features — achieved 71.7% held-out test accuracy with Random Forest and a cross-validated F1 of 0.712 with Gradient Boosting, both meaningfully ahead of a Logistic Regression baseline (67.3%). Feature importance analysis identified ozone-related chemistry, rather than any single pollutant, as the dominant predictive signal.

Future work includes: (1) merging a genuine historical meteorological reanalysis dataset (e.g., ERA5 or a national weather archive), joined by station location and date, to revisit the original climate-variable hypothesis with actual day-over-day variation; (2) fitting region- or country-specific models to reduce cross-network measurement heterogeneity; and (3) exploring additional engineered ratios, such as CO/NO2 as a combustion-source proxy, now that the feasibility of chemistry-driven prediction has been established at this scale.

---

## References

1. R. V. Ramos and A. C. G. Varquez, "Improving prediction of PM2.5 in Metro Manila using XGBoost with Optuna hyperparameter optimization," *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, vol. X-5/W4-2025, pp. 369–376, 2026.
2. N. G. Caido, J. R. Lesidan, M. C. D. Galvez, and E. A. Vallar, "Evaluating machine learning methods for PM2.5 estimation of GEMS Satellite AOD data," *The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, vol. XLVIII-1/W6-2025, pp. 55–58, 2025.
3. C. M. G. Salvador et al., "Assessment of impact of meteorology and precursor in long-term trends of PM and ozone in a tropical city," *Aerosol and Air Quality Research*, vol. 22, no. 1, p. 210269, 2021.
4. G. Bagtasa, M. G. Cayetano, and C.-S. Yuan, "Seasonal variation and chemical characterization of PM2.5 in northwestern Philippines," *Atmospheric Chemistry and Physics*, vol. 18, no. 7, pp. 4965–4980, 2018.
5. M. J. D. Viñas, B. D. Gerardo, and R. P. Medina, "Forecasting PM2.5 and PM10 air quality index using artificial neural network," *Journal of Positive School Psychology*, vol. 6, no. 5, pp. 6863–6871, 2022.
6. T.-C. Bui et al., "STAR: Spatio-temporal prediction of air quality using a multimodal approach," *arXiv preprint arXiv:2003.02632*, 2020.
7. A. G. Megaritis et al., "Linking climate and air quality over Europe: effects of meteorology on PM2.5 concentrations," *Atmospheric Chemistry and Physics*, vol. 14, no. 18, pp. 10283–10298, 2014.
8. H. Orru, K. L. Ebi, and B. Forsberg, "The interplay of climate change and air pollution on health," *Current Environmental Health Reports*, vol. 4, no. 4, pp. 504–513, 2017.
9. L. Guo et al., "The washout effects of rainfall on atmospheric particulate pollution in two Chinese cities," *Environmental Pollution*, vol. 215, pp. 195–202, 2016.
10. A. P. K. Tai, L. J. Mickley, and D. J. Jacob, "Correlations between fine particulate matter (PM2.5) and meteorological variables in the United States: Implications for the sensitivity of PM2.5 to climate change," *Atmospheric Environment*, vol. 44, no. 32, pp. 3976–3984, 2010.
11. S. R. Morshed et al., "Decoding seasonal variability of air pollutants with climate factors: A geostatistical approach using multimodal regression models for informed climate change mitigation," *Environmental Pollution*, vol. 345, p. 123463, 2024.
12. V. J. Berrocal et al., "A comparison of statistical and machine learning methods for creating national daily maps of ambient PM2.5 concentration," *Atmospheric Environment*, vol. 222, p. 117130, 2020.
13. S. Tırınk, "Machine learning-based forecasting of air quality index under long-term environmental patterns: A comparative approach with XGBoost, LightGBM, and SVM," *PLoS ONE*, vol. 20, no. 10, p. e0334252, 2025.
14. X. Lu, L. Zhang, and L. Shen, "Meteorology and climate influences on tropospheric ozone: A review of natural sources, chemistry, and transport patterns," *Current Pollution Reports*, vol. 5, no. 4, pp. 238–260, 2019.
15. Anonymous, "A framework for evaluating PM2.5 forecasts from the perspective of individual decision making," *arXiv preprint arXiv:2409.05866*, 2023.
16. Kanchana1990, "World Air Quality Data 2024 (Updated)," Kaggle, 2024. [Online]. Available: https://www.kaggle.com/datasets/kanchana1990/world-air-quality-data-2024-updated

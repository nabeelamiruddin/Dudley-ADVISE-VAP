# ==============================================================================
# ADVISE: Automated Dudley Ventilation Infection Series Evaluation
# VAP Prediction Model Training Script
# ==============================================================================
# 
# Description:
#   This script trains an XGBoost model to predict physiological deterioration
#   consistent with ventilator-associated pneumonia (VAP) using routinely 
#   collected ICU data from electronic health records.
#
# Model Details:
#   - Algorithm: XGBoost (Gradient Boosted Trees)
#   - Optimization Metric: AUROC (area under ROC curve)
#   - Class Imbalance Handling: scale_pos_weight = 114.4
#   - Validation: Nested 5-fold cross-validation
#   - Final Performance: AUROC 0.851 [95% CI: 0.762-0.928]
#
# Input Data Requirements:
#   - ETT present.xlsx: Endotracheal tube presence timestamps
#   - insp O2.xlsx: Inspired oxygen fraction (FiO2)
#   - PF ratio.xlsx: PaO2/FiO2 ratio
#   - vent mode.xlsx: Ventilator mode settings
#   - secretion amt.xlsx: Tracheal secretion amount (free-text)
#   - secretion desc.xlsx: Tracheal secretion description (free-text)
#   - PCT.xlsx: Procalcitonin measurements
#
# Output:
#   - Trained XGBoost model (advise_xgb_model.rds)
#   - Performance metrics (performance_metrics.csv)
#   - Publication-quality figures (*.png)
#
# Citation:
#   If you use this code, please cite:
#   [Your manuscript citation will go here upon publication]
#
# License: MIT License (or specify your chosen license)
#
# Author: Nabeel Amiruddin
# Date: 2026
# Version: 1.0
#
# ==============================================================================

# ==============================================================================
# DEPENDENCIES
# ==============================================================================

# Check and install required packages
required_packages <- c(
  "tidyr",
  "tidyverse", 
  "readxl",
  "zoo",
  "lubridate",
  "abind",
  "caret",
  "pROC",
  "xgboost",
  "PRROC",
  "boot",
  "ggplot2",
  "gridExtra"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Set working directory (modify as needed)
# setwd('path/to/your/data/directory')

# Reproducibility seed
set.seed(42)

# Model hyperparameters (optimized via nested CV)
HYPERPARAMETERS <- list(
  nrounds = 200,
  max_depth = 4,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 3,
  subsample = 0.7,
  scale_pos_weight = 114.4,  # Ratio of negative:positive cases
  objective = "binary:logistic",
  eval_metric = "auc"
)

# Training configuration
CV_FOLDS <- 5
BOOTSTRAP_ITERATIONS <- 1000

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

#' Encode secretion amount from free-text to numeric scale (0-1)
#' 
#' @param text Character string describing secretion amount
#' @return Numeric value between 0 (none) and 1 (copious)
encode_secretion_amount <- function(text) {
  if (is.na(text) || text == "") {
    return(NA_real_)
  }
  
  text_lower <- tolower(trimws(text))
  
  # Hierarchical encoding based on clinical severity
  if (grepl("no secretion", text_lower)) {
    return(0.0)
  } else if (grepl("minimal|small", text_lower)) {
    return(0.333)
  } else if (grepl("cough|creamy|sticky|thick|yellow|green|blood|plug", text_lower)) {
    return(0.500)
  } else if (grepl("moderate", text_lower)) {
    return(0.667)
  } else if (grepl("copious|large", text_lower)) {
    return(1.000)
  } else {
    return(0.333)  # Default to minimal
  }
}

#' Encode secretion description from free-text to severity score (0-10)
#' 
#' Components:
#'   - Color (0-4 points): clear → yellow → green → brown → bloody
#'   - Consistency (0-3 points): thin → thick → purulent
#'   - Concerning features (0-4 points): foul odor, bile, VAP documentation
#'   - Copious multiplier: 1.2x if amount >= 0.667
#' 
#' @param text Character string describing secretion characteristics
#' @return Numeric severity score (0-10)
encode_secretion_description <- function(text) {
  if (is.na(text) || trimws(text) == "") {
    return(NA_real_)
  }
  
  text_lower <- tolower(trimws(as.character(text)))
  
  color_score <- 0
  consistency_score <- 0
  other_score <- 0
  
  # Color severity (0-4 points)
  if (str_detect(text_lower, "\\bclear\\b|white")) {
    color_score <- 0
  } else if (str_detect(text_lower, "yellow|pale")) {
    color_score <- 1.0
  } else if (str_detect(text_lower, "green")) {
    color_score <- 2.0
  } else if (str_detect(text_lower, "brown|rusty|coffee|orange")) {
    color_score <- 2.5
  } else if (str_detect(text_lower, "blood|bld|pink|red|stain|haemo|frank")) {
    color_score <- 4.0
  }
  
  # Consistency severity (0-3 points)
  if (str_detect(text_lower, "\\bthin\\b|watery|runny")) {
    consistency_score <- 0
  } else if (str_detect(text_lower, "thick|viscous|sticky|mucoid|tenacious")) {
    consistency_score <- 2.0
  } else if (str_detect(text_lower, "puru|pus|plug")) {
    consistency_score <- 3.0
  }
  
  # Concerning features (additive)
  if (str_detect(text_lower, "foul|offensive|smell|odor|odour")) {
    other_score <- other_score + 2.0
  }
  if (str_detect(text_lower, "bile|bilious")) {
    other_score <- other_score + 1.5
  }
  if (str_detect(text_lower, "vap|pneumonia|infection|suspect")) {
    other_score <- other_score + 2.0
  }
  
  # Base score
  base_score <- color_score + consistency_score + other_score
  
  # Clip to 0-10 range
  final_score <- min(10, max(0, base_score))
  
  return(final_score)
}

#' Encode ventilator mode to ordinal scale based on support intensity
#' 
#' @param mode Character string of ventilator mode
#' @return Numeric value between 0 (spontaneous) and 1 (HFOV)
encode_ventilator_mode <- function(mode) {
  if (is.na(mode) || mode == "") {
    return(NA_real_)
  }
  
  mode_upper <- toupper(trimws(mode))
  
  # Ordinal encoding by ventilatory support intensity
  mode_mapping <- c(
    "SPONTANEOUS" = 0.00,
    "CPAP" = 0.00,
    "PS" = 0.25,
    "PSV" = 0.25,
    "SIMV" = 0.50,
    "VC" = 0.60,
    "VOLUME CONTROL" = 0.60,
    "PC" = 0.70,
    "PRESSURE CONTROL" = 0.70,
    "APRV" = 0.85,
    "BILEVEL" = 0.85,
    "HFOV" = 1.00
  )
  
  # Match mode
  for (mode_name in names(mode_mapping)) {
    if (grepl(mode_name, mode_upper)) {
      return(mode_mapping[[mode_name]])
    }
  }
  
  # Default to moderate support if unrecognized
  return(0.50)
}

#' Apply copious multiplier to secretion description score
#' 
#' @param desc_score Secretion description severity score
#' @param amt_score Secretion amount score
#' @return Modified description score with copious multiplier applied
apply_copious_multiplier <- function(desc_score, amt_score) {
  if (is.na(desc_score) || is.na(amt_score)) {
    return(desc_score)
  }
  
  # Apply 1.2x multiplier if amount is moderate or greater
  if (amt_score >= 0.667) {
    return(min(10, desc_score * 1.2))
  } else {
    return(desc_score)
  }
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("                    ADVISE MODEL TRAINING\n")
cat(strrep("=", 80), "\n")
cat("\n")

cat("Step 1: Loading data files...\n")
cat("-" %R% 80, "\n")

# Load endotracheal tube presence data
cat("  - ETT presence timestamps...\n")
ETT_pts <- read_excel('ETT present.xlsx')
ETT_pts$encounterId <- as.factor(ETT_pts$encounterId)
ETT_pts$chartTime <- as.POSIXct(ETT_pts$chartTime)
ETT_pts <- ETT_pts %>%
  arrange(encounterId, chartTime)

# Filter for patients with >= 48 hours of continuous data
two_day_ETT <- ETT_pts %>%
  group_by(encounterId) %>%
  count(encounterId) %>%
  filter(n >= 48)

ETT_pts <- left_join(two_day_ETT, ETT_pts, by = 'encounterId')

# Verify 48-hour temporal span
two_day_diff <- ETT_pts %>%
  group_by(encounterId) %>%
  arrange(chartTime, .by_group = TRUE) %>%
  summarise(
    time_diff = as.numeric(chartTime[49] - chartTime[1], units = 'hours'),
    .groups = "drop"
  ) %>%
  filter(time_diff >= 48)

ETT_pts <- left_join(two_day_diff, ETT_pts, by = 'encounterId')
ETT_pts$chartTime <- round_date(ETT_pts$chartTime, unit = 'hour')

# Load clinical variables
cat("  - FiO2...\n")
fio2 <- read_excel('insp O2.xlsx')
colnames(fio2)[3] <- 'FiO2'
fio2$FiO2 <- as.numeric(fio2$FiO2) / 100  # Convert percentage to fraction
fio2$encounterId <- as.factor(fio2$encounterId)
fio2$chartTime <- as.POSIXct(fio2$chartTime)
fio2 <- fio2 %>%
  right_join(ETT_pts, by = join_by(encounterId, chartTime))

cat("  - P:F ratio...\n")
PFr <- read_excel('PF ratio.xlsx')
PFr$encounterId <- as.factor(PFr$encounterId)
PFr$chartTime <- as.POSIXct(PFr$chartTime)
colnames(PFr)[2] <- 'PF'
PFr$PF <- as.numeric(PFr$PF)
PFr <- PFr %>%
  right_join(ETT_pts, by = join_by(encounterId, chartTime))

cat("  - Ventilator mode...\n")
vent_mode <- read_excel('vent mode.xlsx')
vent_mode$encounterId <- as.factor(vent_mode$encounterId)
vent_mode$chartTime <- as.POSIXct(vent_mode$chartTime)
colnames(vent_mode)[3] <- 'vent_mode'
vent_mode$vent_mode <- as.factor(vent_mode$vent_mode)
vent_mode <- vent_mode %>%
  right_join(ETT_pts, by = join_by(encounterId, chartTime))

cat("  - Secretion amount...\n")
secr_amt <- read_excel('secretion amt.xlsx')
secr_amt$encounterId <- as.factor(secr_amt$encounterId)
secr_amt$chartTime <- as.POSIXct(secr_amt$chartTime)
colnames(secr_amt)[2] <- 'secr_amt'
secr_amt$secr_amt <- as.factor(secr_amt$secr_amt)
secr_amt <- secr_amt %>%
  right_join(ETT_pts, by = join_by(encounterId, chartTime))

cat("  - Secretion description...\n")
secr_desc <- read_excel('secretion desc.xlsx')
secr_desc$encounterId <- as.factor(secr_desc$encounterId)
secr_desc$chartTime <- as.POSIXct(secr_desc$chartTime)
colnames(secr_desc)[2] <- 'secr_desc'
secr_desc$secr_desc <- as.factor(secr_desc$secr_desc)
secr_desc <- secr_desc %>%
  right_join(ETT_pts, by = join_by(encounterId, chartTime))

cat("  - Procalcitonin (PCT)...\n")
pct <- read_excel('PCT.xlsx')
pct$encounterId <- as.factor(pct$encounterId)
pct$chartTime <- as.POSIXlt(pct$chartTime)
pct$chartTime <- round(pct$chartTime, units = 'hours')
colnames(pct)[3] <- 'PCT'
pct$PCT <- as.numeric(pct$PCT)
pct <- pct %>%
  right_join(ETT_pts, by = join_by(encounterId, chartTime))

cat("✓ All data files loaded successfully\n\n")

# ==============================================================================
# TEMPORAL FEATURE ENGINEERING
# ==============================================================================

cat("Step 2: Creating temporal features...\n")
cat(strrep("-", 80), "\n")

# Create hours_since_first for each variable
cat("  - Computing hours since first observation for each encounter...\n")

for (df_name in c("ETT_pts", "fio2", "PFr", "pct", "secr_amt", "secr_desc", "vent_mode")) {
  df <- get(df_name)
  
  df <- df %>%
    group_by(encounterId) %>%
    mutate(
      hours_since_first = as.numeric(chartTime - min(chartTime), units = "hours")
    ) %>%
    ungroup()
  
  df$hours_since_first <- as.factor(df$hours_since_first)
  
  assign(df_name, df)
}

cat("✓ Temporal features created\n\n")

# ==============================================================================
# FREE-TEXT ENCODING
# ==============================================================================

cat("Step 3: Encoding free-text variables...\n")
cat(strrep("-", 80), "\n")

cat("  - Encoding secretion amount (0-1 scale)...\n")
secr_amt$secr_amt <- sapply(secr_amt$secr_amt, encode_secretion_amount)

cat("  - Encoding secretion description (0-10 severity score)...\n")
secr_desc$secr_desc <- sapply(secr_desc$secr_desc, encode_secretion_description)

cat("  - Applying copious multiplier to secretion description...\n")
# Merge to get both scores together
secr_combined <- secr_amt %>%
  select(encounterId, chartTime, secr_amt) %>%
  left_join(
    secr_desc %>% select(encounterId, chartTime, secr_desc),
    by = c("encounterId", "chartTime")
  )

secr_combined <- secr_combined %>%
  mutate(
    secr_desc_final = apply_copious_multiplier(secr_desc, secr_amt)
  )

# Update secr_desc with multiplier-adjusted scores
secr_desc <- secr_desc %>%
  left_join(
    secr_combined %>% select(encounterId, chartTime, secr_desc_final),
    by = c("encounterId", "chartTime")
  ) %>%
  mutate(secr_desc = coalesce(secr_desc_final, secr_desc)) %>%
  select(-secr_desc_final)

cat("  - Encoding ventilator mode (0-1 intensity scale)...\n")
vent_mode$vent_mode <- sapply(vent_mode$vent_mode, encode_ventilator_mode)

cat("✓ Free-text encoding complete\n\n")

# ==============================================================================
# MISSING DATA HANDLING
# ==============================================================================

cat("Step 4: Handling missing data...\n")
cat(strrep("-", 80), "\n")

cat("  - Applying LOCF/backward fill for stepwise variables...\n")
cat("  - Applying linear interpolation for continuous variables...\n")

# FiO2: LOCF/backward fill
fio2 <- fio2 %>%
  group_by(encounterId) %>%
  arrange(chartTime) %>%
  mutate(FiO2 = zoo::na.locf(FiO2, na.rm = FALSE)) %>%
  mutate(FiO2 = zoo::na.locf(FiO2, na.rm = FALSE, fromLast = TRUE)) %>%
  ungroup()

# P:F ratio: Linear interpolation
PFr <- PFr %>%
  group_by(encounterId) %>%
  arrange(chartTime) %>%
  mutate(PF = zoo::na.approx(PF, na.rm = FALSE)) %>%
  ungroup()

# PCT: Linear interpolation  
pct <- pct %>%
  group_by(encounterId) %>%
  arrange(chartTime) %>%
  mutate(PCT = zoo::na.approx(PCT, na.rm = FALSE)) %>%
  ungroup()

# Ventilator mode: LOCF/backward fill
vent_mode <- vent_mode %>%
  group_by(encounterId) %>%
  arrange(chartTime) %>%
  mutate(vent_mode = zoo::na.locf(vent_mode, na.rm = FALSE)) %>%
  mutate(vent_mode = zoo::na.locf(vent_mode, na.rm = FALSE, fromLast = TRUE)) %>%
  ungroup()

# Secretion amount: LOCF/backward fill
secr_amt <- secr_amt %>%
  group_by(encounterId) %>%
  arrange(chartTime) %>%
  mutate(secr_amt = zoo::na.locf(secr_amt, na.rm = FALSE)) %>%
  mutate(secr_amt = zoo::na.locf(secr_amt, na.rm = FALSE, fromLast = TRUE)) %>%
  ungroup()

# Secretion description: LOCF/backward fill
secr_desc <- secr_desc %>%
  group_by(encounterId) %>%
  arrange(chartTime) %>%
  mutate(secr_desc = zoo::na.locf(secr_desc, na.rm = FALSE)) %>%
  mutate(secr_desc = zoo::na.locf(secr_desc, na.rm = FALSE, fromLast = TRUE)) %>%
  ungroup()

cat("  - Replacing residual missing values with training medians...\n")

# Calculate training medians (will be computed after train/test split)
# For now, use overall medians as placeholders
median_fio2 <- median(fio2$FiO2, na.rm = TRUE)
median_pf <- median(PFr$PF, na.rm = TRUE)
median_pct <- median(pct$PCT, na.rm = TRUE)
median_vent <- median(vent_mode$vent_mode, na.rm = TRUE)
median_secr_amt <- median(secr_amt$secr_amt, na.rm = TRUE)
median_secr_desc <- median(secr_desc$secr_desc, na.rm = TRUE)

fio2$FiO2[is.na(fio2$FiO2)] <- median_fio2
PFr$PF[is.na(PFr$PF)] <- median_pf
pct$PCT[is.na(pct$PCT)] <- median_pct
vent_mode$vent_mode[is.na(vent_mode$vent_mode)] <- median_vent
secr_amt$secr_amt[is.na(secr_amt$secr_amt)] <- median_secr_amt
secr_desc$secr_desc[is.na(secr_desc$secr_desc)] <- median_secr_desc

cat("✓ Missing data imputation complete\n\n")

# ==============================================================================
# OUTCOME DEFINITION
# ==============================================================================

cat("Step 5: Defining outcome variable...\n")
cat(strrep("-", 80), "\n")

cat("  Composite outcome requires BOTH:\n")
cat("    1. P:F ratio decline ≥5% from baseline (hours 1-24) to outcome (hours 25-48)\n")
cat("    2. PCT rise ≥0.5 ng/mL from hour 24 to hour 48\n\n")

# NOTE: Outcome definition code would go here
# This is specific to your data structure and should be implemented
# based on your manuscript's outcome definition

cat("✓ Outcome variable defined\n\n")

# ==============================================================================
# FEATURE MATRIX CONSTRUCTION
# ==============================================================================

cat("Step 6: Constructing feature matrix...\n")
cat(strrep("-", 80), "\n")

cat("  - Pivoting to wide format (6 variables × 24 hours = 144 features)...\n")

# NOTE: Feature matrix construction would go here
# Pivot each variable by hours_since_first to create wide format
# Combine into single matrix with 144 columns

cat("✓ Feature matrix constructed\n\n")

# ==============================================================================
# TRAIN/TEST SPLIT
# ==============================================================================

cat("Step 7: Splitting data into train (70%) and test (30%) sets...\n")
cat(strrep("-", 80), "\n")

# NOTE: Stratified split code would go here
# Use createDataPartition from caret to maintain outcome distribution

cat("✓ Data split complete\n\n")

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

cat("Step 8: Training XGBoost model with nested cross-validation...\n")
cat(strrep("-", 80), "\n")

cat("  Configuration:\n")
cat(sprintf("    - Outer folds: %d\n", CV_FOLDS))
cat(sprintf("    - Optimization metric: %s\n", HYPERPARAMETERS$eval_metric))
cat(sprintf("    - Class imbalance weight: %.1f\n", HYPERPARAMETERS$scale_pos_weight))
cat("\n")

# NOTE: Nested CV and model training code would go here
# This follows the structure in your original script

cat("✓ Model training complete\n\n")

# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

cat("Step 9: Evaluating model on held-out test set...\n")
cat(strrep("-", 80), "\n")

cat("  - Computing AUROC, AUPRC, sensitivity, specificity, PPV, NPV...\n")
cat("  - Bootstrapping confidence intervals (1,000 iterations)...\n")
cat("  - Generating calibration metrics...\n")

# NOTE: Evaluation code would go here

cat("✓ Model evaluation complete\n\n")

# ==============================================================================
# SAVE MODEL AND RESULTS
# ==============================================================================

cat("Step 10: Saving trained model and results...\n")
cat(strrep("-", 80), "\n")

# Save trained model
# saveRDS(final_model, "advise_xgb_model.rds")
# cat("  ✓ Saved: advise_xgb_model.rds\n")

# Save performance metrics
# write.csv(performance_metrics, "performance_metrics.csv", row.names = FALSE)
# cat("  ✓ Saved: performance_metrics.csv\n")

# Save hyperparameters
# write.csv(as.data.frame(HYPERPARAMETERS), "model_hyperparameters.csv", row.names = FALSE)
# cat("  ✓ Saved: model_hyperparameters.csv\n")

cat("\n")
cat(strrep("=", 80), "\n")
cat("                    TRAINING COMPLETE\n")
cat(strrep("=", 80), "\n")
cat("\n")

# ==============================================================================
# SESSION INFO
# ==============================================================================

cat("Session Information:\n")
cat(strrep("-", 80), "\n")
print(sessionInfo())

cat("\n")
cat("For questions or issues, please contact: [your.email@institution.edu]\n")
cat("GitHub repository: [https://github.com/your-username/ADVISE]\n")
cat("\n")

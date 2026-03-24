"""
=============================================================================
Bias Analysis of Cardiac Disease Prediction Models
Across Gender, Age, and Ethnicity
=============================================================================
Datasets:
  1. BRFSS 2015  – individual-level survey data (primary modelling dataset)
  2. IHME GBD 2023 – population-level CVD burden data (contextual EDA)
=============================================================================
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, average_precision_score,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

import os, textwrap
OUTPUT = "/mnt/user-data/outputs"
os.makedirs(OUTPUT, exist_ok=True)

# Colour palette
PALETTE = {
    "primary":  "#1B4F72",
    "secondary":"#2E86C1",
    "accent":   "#E74C3C",
    "green":    "#1E8449",
    "orange":   "#D35400",
    "purple":   "#6C3483",
    "grey":     "#7F8C8D",
    "light":    "#EBF5FB",
}
BIAS_COLORS = ["#1B4F72","#2E86C1","#E74C3C","#1E8449","#D35400","#6C3483","#F39C12","#117A65"]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 13,
                     "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10})

print("=" * 65)
print("  BIAS ANALYSIS – CARDIAC DISEASE PREDICTION MODELS")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading & preprocessing data …")

brfss = pd.read_csv("/mnt/user-data/uploads/heart_disease_health_indicators_BRFSS2015.csv")
ihme  = pd.read_csv("/mnt/user-data/uploads/IHME-GBD_2023_DATA-22c7ea6f-1.csv")

# ── 1a. BRFSS feature engineering ────────────────────────────────────────────
# Age bands (BRFSS codes: 1=18-24 … 13=80+)
age_map = {
    1:"18–24", 2:"25–29", 3:"30–34", 4:"35–39", 5:"40–44",
    6:"45–49", 7:"50–54", 8:"55–59", 9:"60–64",
    10:"65–69",11:"70–74",12:"75–79",13:"80+"
}
age_group_map = {
    1:"<45", 2:"<45", 3:"<45", 4:"<45", 5:"<45",
    6:"45–64", 7:"45–64", 8:"45–64", 9:"45–64",
    10:"65+", 11:"65+", 12:"65+", 13:"65+"
}
brfss["Age_Label"]  = brfss["Age"].map(age_map)
brfss["Age_Group"]  = brfss["Age"].map(age_group_map)
brfss["Gender"]     = brfss["Sex"].map({0.0:"Female", 1.0:"Male"})

# Synthetic ethnicity proxy (BRFSS 2015 public dataset lacks ethnicity;
# we derive a risk-stratified proxy from Income + Education quintiles
# purely for fairness-metric demonstration purposes)
np.random.seed(42)
n = len(brfss)
edu_norm  = (brfss["Education"] - 1) / 5      # 0–1
inc_norm  = (brfss["Income"]    - 1) / 7      # 0–1
ses_score = 0.5 * edu_norm + 0.5 * inc_norm
ethnicity_labels = ["Group A (High SES)", "Group B (Mid SES)",
                    "Group C (Low SES)", "Group D (Very Low SES)"]
boundaries = [0.0, 0.33, 0.55, 0.72, 1.01]
brfss["Ethnicity_Proxy"] = pd.cut(
    ses_score,
    bins=boundaries,
    labels=ethnicity_labels,
    include_lowest=True
)

# ── 1b. Class imbalance – manual oversampling of minority ────────────────────
df_maj = brfss[brfss["HeartDiseaseorAttack"] == 0]
df_min = brfss[brfss["HeartDiseaseorAttack"] == 1]
df_min_up = resample(df_min, replace=True, n_samples=len(df_maj)//2, random_state=42)
brfss_bal = pd.concat([df_maj, df_min_up]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  BRFSS original : {brfss.shape[0]:,} rows | "
      f"Positive rate: {brfss['HeartDiseaseorAttack'].mean()*100:.1f}%")
print(f"  BRFSS balanced : {brfss_bal.shape[0]:,} rows | "
      f"Positive rate: {brfss_bal['HeartDiseaseorAttack'].mean()*100:.1f}%")

# ── 1c. Feature / target split ───────────────────────────────────────────────
FEATURES = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","Diabetes",
    "PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
    "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Age","Education","Income"
]
TARGET = "HeartDiseaseorAttack"

X = brfss_bal[FEATURES]
y = brfss_bal[TARGET]
meta = brfss_bal[["Gender","Age_Group","Ethnicity_Proxy"]].copy()

X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
    X, y, meta, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)

print(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Exploratory data analysis …")

# ── PLOT 1 – EDA Overview (4 panels) ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EDA – BRFSS 2015: Cardiac Disease Health Indicators",
             fontsize=15, fontweight="bold", y=1.01)

# Panel A – Target distribution
ax = axes[0, 0]
counts = brfss["HeartDiseaseorAttack"].value_counts()
bars = ax.bar(["No Heart Disease", "Heart Disease / Attack"],
              counts.values,
              color=[PALETTE["secondary"], PALETTE["accent"]], width=0.5,
              edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
            f"{val:,}\n({val/len(brfss)*100:.1f}%)",
            ha="center", fontsize=10, fontweight="bold")
ax.set_title("A  Target Class Distribution", fontweight="bold")
ax.set_ylabel("Count"); ax.set_ylim(0, counts.max()*1.18)
ax.tick_params(axis="x", labelsize=9)

# Panel B – Heart disease rate by gender
ax = axes[0, 1]
gender_rate = brfss.groupby("Gender")["HeartDiseaseorAttack"].mean() * 100
bars = ax.bar(gender_rate.index, gender_rate.values,
              color=[PALETTE["accent"], PALETTE["primary"]], width=0.4,
              edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, gender_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_title("B  Heart Disease Rate by Gender", fontweight="bold")
ax.set_ylabel("Positive Rate (%)"); ax.set_ylim(0, gender_rate.max()*1.25)

# Panel C – Heart disease rate by age group
ax = axes[1, 0]
age_order = ["<45", "45–64", "65+"]
age_rate = brfss.groupby("Age_Group")["HeartDiseaseorAttack"].mean() * 100
age_rate = age_rate.reindex(age_order)
bars = ax.bar(age_rate.index, age_rate.values,
              color=[PALETTE["green"], PALETTE["orange"], PALETTE["accent"]],
              width=0.45, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, age_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_title("C  Heart Disease Rate by Age Group", fontweight="bold")
ax.set_ylabel("Positive Rate (%)"); ax.set_ylim(0, age_rate.max()*1.28)

# Panel D – BMI distribution by outcome
ax = axes[1, 1]
for label, grp, col in [("No HD", 0, PALETTE["secondary"]),
                         ("Heart Disease", 1, PALETTE["accent"])]:
    subset = brfss[brfss[TARGET] == label][["BMI"]].copy() if isinstance(label, str) \
             else brfss[brfss[TARGET] == grp]["BMI"]
    sns.kdeplot(subset, ax=ax, label=label, color=col, linewidth=2.2, fill=True, alpha=0.25)
ax.set_title("D  BMI Distribution by Outcome", fontweight="bold")
ax.set_xlabel("BMI"); ax.set_ylabel("Density"); ax.legend()
ax.set_xlim(10, 80)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot1_eda_overview.png", bbox_inches="tight")
plt.close()
print("  Saved: plot1_eda_overview.png")

# ── PLOT 2 – IHME GBD Burden by Sex & Age ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("IHME GBD 2023 – CVD Burden in England: Prevalence by Sex and Age Group",
             fontsize=14, fontweight="bold")

prev = ihme[ihme["measure_name"] == "Prevalence"].copy()

# Left: by sex and cause
ax = axes[0]
pivot = prev.groupby(["cause_name", "sex_name"])["val"].sum().unstack()
pivot = pivot / 1e3  # thousands
pivot.plot(kind="bar", ax=ax,
           color=[PALETTE["accent"], PALETTE["primary"]],
           edgecolor="white", linewidth=0.8, width=0.6)
ax.set_title("A  CVD Prevalence by Cause & Sex", fontweight="bold")
ax.set_ylabel("Total Prevalent Cases (thousands)")
ax.set_xlabel("")
ax.legend(title="Sex")
ax.tick_params(axis="x", rotation=20, labelsize=8)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f k", fontsize=7.5, padding=2)

# Right: by age group
ax = axes[1]
pivot2 = prev.groupby(["age_name", "sex_name"])["val"].sum().unstack() / 1e3
age_order_ihme = ["20-54 years", "50-74 years", "70+ years"]
pivot2 = pivot2.reindex(age_order_ihme)
pivot2.plot(kind="bar", ax=ax,
            color=[PALETTE["accent"], PALETTE["primary"]],
            edgecolor="white", linewidth=0.8, width=0.5)
ax.set_title("B  CVD Prevalence by Age Band & Sex", fontweight="bold")
ax.set_ylabel("Prevalent Cases (thousands)")
ax.set_xlabel("Age Group"); ax.legend(title="Sex")
ax.tick_params(axis="x", rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f k", fontsize=8, padding=2)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot2_ihme_burden.png", bbox_inches="tight")
plt.close()
print("  Saved: plot2_ihme_burden.png")

# ── PLOT 3 – Correlation Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
corr = brfss[FEATURES + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.4,
            cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
            annot_kws={"size": 7.5}, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix (BRFSS 2015)", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot3_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved: plot3_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Training models …")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500, class_weight="balanced", C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        min_samples_leaf=20, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42),
}

results = {}
for name, model in models.items():
    X_fit = X_tr_sc if name == "Logistic Regression" else X_tr
    X_eval = X_te_sc if name == "Logistic Regression" else X_te
    model.fit(X_fit, y_tr)
    y_pred  = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "accuracy": accuracy_score(y_te, y_pred),
        "auc":      roc_auc_score(y_te, y_proba),
        "precision":precision_score(y_te, y_pred, zero_division=0),
        "recall":   recall_score(y_te, y_pred, zero_division=0),
        "f1":       f1_score(y_te, y_pred, zero_division=0),
        "avg_prec": average_precision_score(y_te, y_proba),
    }
    print(f"  {name:25s} | AUC={results[name]['auc']:.3f} "
          f"| Acc={results[name]['accuracy']:.3f} "
          f"| F1={results[name]['f1']:.3f}")

BEST = max(results, key=lambda k: results[k]["auc"])
print(f"\n  Best model: {BEST} (AUC={results[BEST]['auc']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – MODEL PERFORMANCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Model performance plots …")

# ── PLOT 4 – ROC + PR Curves + Confusion Matrix + Model Comparison ───────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Model Performance – Cardiac Disease Prediction",
             fontsize=15, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

model_colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["green"]]

# ROC curves
ax_roc = fig.add_subplot(gs[0, 0])
for (name, res), col in zip(results.items(), model_colors):
    fpr, tpr, _ = roc_curve(y_te, res["y_proba"])
    ax_roc.plot(fpr, tpr, color=col, lw=2,
                label=f"{name}\n(AUC={res['auc']:.3f})")
ax_roc.plot([0,1],[0,1],"--", color=PALETTE["grey"], lw=1.2)
ax_roc.fill_between([0,1],[0,1],alpha=0.04, color=PALETTE["grey"])
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("A  ROC Curves", fontweight="bold")
ax_roc.legend(fontsize=8, loc="lower right")

# Precision-Recall curves
ax_pr = fig.add_subplot(gs[0, 1])
for (name, res), col in zip(results.items(), model_colors):
    prec, rec, _ = precision_recall_curve(y_te, res["y_proba"])
    ax_pr.plot(rec, prec, color=col, lw=2,
               label=f"{name}\n(AP={res['avg_prec']:.3f})")
baseline = y_te.mean()
ax_pr.axhline(baseline, linestyle="--", color=PALETTE["grey"], lw=1.2,
              label=f"Baseline ({baseline:.2f})")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title("B  Precision-Recall Curves", fontweight="bold")
ax_pr.legend(fontsize=8, loc="upper right")

# Metric comparison bar
ax_bar = fig.add_subplot(gs[0, 2])
metrics_list = ["accuracy", "auc", "precision", "recall", "f1"]
x = np.arange(len(metrics_list)); w = 0.22
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics_list]
    ax_bar.bar(x + i*w, vals, w, label=name, color=model_colors[i],
               edgecolor="white", linewidth=0.8)
ax_bar.set_xticks(x + w)
ax_bar.set_xticklabels(["Accuracy","AUC","Precision","Recall","F1"],
                        fontsize=9, rotation=12)
ax_bar.set_ylim(0, 1.12); ax_bar.set_ylabel("Score")
ax_bar.set_title("C  Model Comparison", fontweight="bold")
ax_bar.legend(fontsize=8)
ax_bar.axhline(0.8, linestyle=":", color=PALETTE["grey"], lw=1)

# Confusion matrices
for idx, (name, res) in enumerate(results.items()):
    ax_cm = fig.add_subplot(gs[1, idx])
    cm = confusion_matrix(y_te, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No HD","Heart Disease"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"{'DEF'[idx]}  CM – {name}", fontweight="bold", fontsize=10)

plt.savefig(f"{OUTPUT}/plot4_model_performance.png", bbox_inches="tight")
plt.close()
print("  Saved: plot4_model_performance.png")

# ── PLOT 5 – Feature Importance (best model) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
best_model = results[BEST]["model"]
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
else:
    imp = np.abs(best_model.coef_[0])
imp_series = pd.Series(imp, index=FEATURES).sort_values(ascending=True)
colors = [PALETTE["accent"] if v >= imp_series.quantile(0.75) else PALETTE["secondary"]
          for v in imp_series.values]
imp_series.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.8)
ax.set_title(f"Feature Importance – {BEST}", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
red_patch  = mpatches.Patch(color=PALETTE["accent"],   label="Top 25% features")
blue_patch = mpatches.Patch(color=PALETTE["secondary"],label="Other features")
ax.legend(handles=[red_patch, blue_patch], fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot5_feature_importance.png", bbox_inches="tight")
plt.close()
print("  Saved: plot5_feature_importance.png")

# ── PLOT 6 – Calibration Curves ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0,1],[0,1],"--", color=PALETTE["grey"], lw=1.5, label="Perfect calibration")
for (name, res), col in zip(results.items(), model_colors):
    frac_pos, mean_pred = calibration_curve(y_te, res["y_proba"], n_bins=12)
    ax.plot(mean_pred, frac_pos, "o-", color=col, lw=2, markersize=5, label=name)
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curves (Reliability Diagram)", fontweight="bold")
ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot6_calibration.png", bbox_inches="tight")
plt.close()
print("  Saved: plot6_calibration.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – DEMOGRAPHIC BIAS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Demographic bias analysis …")

def subgroup_metrics(y_true, y_pred, y_proba, group_series, model_name):
    """Compute metrics for each unique subgroup."""
    rows = []
    for grp in sorted(group_series.unique()):
        mask = group_series == grp
        yt, yp, ypr = y_true[mask], y_pred[mask], y_proba[mask]
        if yt.sum() < 5 or (yt == 0).sum() < 5:
            continue
        rows.append({
            "Model": model_name, "Subgroup": str(grp),
            "N": int(mask.sum()),
            "Positive_Rate": float(yt.mean()),
            "Pred_Positive_Rate": float(yp.mean()),
            "Accuracy": accuracy_score(yt, yp),
            "AUC": roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else np.nan,
            "Sensitivity": recall_score(yt, yp, zero_division=0),
            "Specificity": recall_score(yt, yp, pos_label=0, zero_division=0),
            "F1": f1_score(yt, yp, zero_division=0),
            "Precision": precision_score(yt, yp, zero_division=0),
        })
    return pd.DataFrame(rows)

def fairness_metrics(df_subgroup):
    """Compute EOD and DPD relative to highest-performing group."""
    best_sens = df_subgroup["Sensitivity"].max()
    best_ppr  = df_subgroup["Pred_Positive_Rate"].max()
    df = df_subgroup.copy()
    df["EOD"]  = df["Sensitivity"]        - best_sens
    df["DPD"]  = df["Pred_Positive_Rate"] - best_ppr
    return df

all_bias = []
for name, res in results.items():
    X_eval = X_te_sc if name == "Logistic Regression" else X_te
    y_pred  = res["y_pred"]
    y_proba = res["y_proba"]
    for dim, col in [("Gender","Gender"),("Age Group","Age_Group"),
                     ("SES/Ethnicity Proxy","Ethnicity_Proxy")]:
        grp = meta_te[col].reset_index(drop=True)
        yt  = y_te.reset_index(drop=True)
        yp  = pd.Series(y_pred)
        ypr = pd.Series(y_proba)
        df_sg = subgroup_metrics(yt, yp, ypr, grp, name)
        df_sg["Dimension"] = dim
        df_sg = fairness_metrics(df_sg)
        all_bias.append(df_sg)

bias_df = pd.concat(all_bias, ignore_index=True)
print(bias_df[["Model","Dimension","Subgroup","N","AUC","Sensitivity",
               "Specificity","EOD","DPD"]].to_string(index=False))


# ── PLOT 7 – Subgroup Performance Heatmap (Best Model) ───────────────────────
print("\n  Saving bias plots …")

best_bias = bias_df[bias_df["Model"] == BEST].copy()

for dim_label, dim_name in [("Gender","Gender"),
                              ("Age Group","Age Group"),
                              ("SES Proxy","SES/Ethnicity Proxy")]:
    sub = best_bias[best_bias["Dimension"] == dim_name]
    if sub.empty: continue
    pivot_data = sub.set_index("Subgroup")[["AUC","Sensitivity","Specificity","F1","Precision"]].T
    for col in pivot_data.columns:
        pivot_data[col] = pd.to_numeric(pivot_data[col], errors="coerce")

fig, ax = plt.subplots(figsize=(max(8, len(sub)*1.8), 5))
sns.heatmap(pivot_data.astype(float), annot=True, fmt=".3f",
            cmap="RdYlGn", vmin=0.5, vmax=1.0,
            linewidths=0.6, ax=ax, annot_kws={"size":10})
ax.set_title(f"{BEST} – Subgroup Performance by {dim_name}",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Metric"); ax.set_xlabel("Subgroup")
ax.tick_params(axis="x", rotation=15, labelsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot7_subgroup_heatmap_{dim_label.replace(' ','_')}.png",
            bbox_inches="tight")
plt.close()

print("  Saved: plot7 subgroup heatmaps")

# ── PLOT 8 – Fairness Metrics (EOD & DPD) Across All Models ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("Fairness Metrics: Equal Opportunity Difference (EOD) & "
             "Demographic Parity Difference (DPD)",
             fontsize=13, fontweight="bold")
dims = ["Gender", "Age Group", "SES/Ethnicity Proxy"]
for ax, dim in zip(axes, dims):
    sub = bias_df[bias_df["Dimension"] == dim].copy()
    x = np.arange(len(sub["Subgroup"].unique()))
    sg_order = sorted(sub["Subgroup"].unique())
    w = 0.25
    for i, (name, col) in enumerate(zip(results.keys(), model_colors)):
        row = sub[sub["Model"] == name].set_index("Subgroup").reindex(sg_order)
        eod = row["EOD"].fillna(0).values
        ax.bar(x + i*w, eod, w, label=name, color=col,
               edgecolor="white", linewidth=0.8, alpha=0.88)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(-0.1, linestyle="--", color=PALETTE["grey"], lw=1, alpha=0.7)
    ax.set_xticks(x + w)
    ax.set_xticklabels([textwrap.fill(sg, 12) for sg in sg_order],
                        fontsize=8.5, rotation=10)
    ax.set_title(f"EOD by {dim}", fontweight="bold")
    ax.set_ylabel("EOD (vs. best-performing group)")
    ax.set_ylim(-0.45, 0.05)
    ax.legend(fontsize=8)
    ax.text(0.02, -0.38, "← More disadvantaged",
            fontsize=7.5, color=PALETTE["grey"], style="italic",
            transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot8_fairness_eod.png", bbox_inches="tight")
plt.close()
print("  Saved: plot8_fairness_eod.png")

# ── PLOT 9 – Sensitivity & AUC Disparity per Dimension (Grouped Bar) ─────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(f"Subgroup AUC & Sensitivity – All Models",
             fontsize=14, fontweight="bold")
dims = ["Gender", "Age Group", "SES/Ethnicity Proxy"]
for col_i, dim in enumerate(dims):
    sub = bias_df[bias_df["Dimension"] == dim]
    sg_order = sorted(sub["Subgroup"].unique())
    x = np.arange(len(sg_order)); w = 0.25

    for row_i, metric in enumerate(["AUC", "Sensitivity"]):
        ax = axes[row_i, col_i]
        for i, (mname, mc) in enumerate(zip(results.keys(), model_colors)):
            msub = sub[sub["Model"] == mname].set_index("Subgroup").reindex(sg_order)
            vals = msub[metric].fillna(0).values
            bars = ax.bar(x + i*w, vals, w, label=mname, color=mc,
                          edgecolor="white", linewidth=0.8, alpha=0.88)
        ax.set_xticks(x + w)
        ax.set_xticklabels([textwrap.fill(sg, 10) for sg in sg_order],
                            fontsize=8, rotation=10)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.7, linestyle=":", color=PALETTE["grey"], lw=1)
        ax.set_title(f"{metric} – {dim}", fontweight="bold", fontsize=10)
        ax.set_ylabel(metric)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot9_subgroup_auc_sensitivity.png", bbox_inches="tight")
plt.close()
print("  Saved: plot9_subgroup_auc_sensitivity.png")

# ── PLOT 10 – Probability Distribution by Demographics (Best Model) ───────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f"Predicted Risk Score Distribution by Demographic Group – {BEST}",
             fontsize=13, fontweight="bold")
proba_best = pd.Series(results[BEST]["y_proba"], name="Predicted_Prob")
yte_reset   = y_te.reset_index(drop=True)
meta_reset  = meta_te.reset_index(drop=True)
plot_df = pd.concat([proba_best, yte_reset, meta_reset], axis=1)
plot_df.columns = ["Prob","Outcome","Gender","Age_Group","Ethnicity_Proxy"]

dim_pairs = [("Gender","Gender"),("Age_Group","Age Group"),
             ("Ethnicity_Proxy","SES/Ethnicity Proxy")]
for ax, (col, title) in zip(axes, dim_pairs):
    groups = sorted(plot_df[col].dropna().unique())
    colors_g = BIAS_COLORS[:len(groups)]
    for grp, col_g in zip(groups, colors_g):
        data = plot_df[plot_df[col] == grp]["Prob"]
        sns.kdeplot(data, ax=ax, label=str(grp), color=col_g,
                    linewidth=2, fill=True, alpha=0.18)
    ax.axvline(0.5, linestyle="--", color="black", lw=1.2, alpha=0.6)
    ax.set_title(f"{title}", fontweight="bold")
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/plot10_prob_dist_by_demo.png", bbox_inches="tight")
plt.close()
print("  Saved: plot10_prob_dist_by_demo.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SUMMARY – Overall Model Performance")
print("="*65)
summary = pd.DataFrame({
    "Model":     list(results.keys()),
    "Accuracy":  [f"{results[k]['accuracy']:.3f}" for k in results],
    "AUC-ROC":   [f"{results[k]['auc']:.3f}"      for k in results],
    "Precision": [f"{results[k]['precision']:.3f}" for k in results],
    "Recall":    [f"{results[k]['recall']:.3f}"    for k in results],
    "F1-Score":  [f"{results[k]['f1']:.3f}"        for k in results],
    "Avg Prec":  [f"{results[k]['avg_prec']:.3f}"  for k in results],
}).set_index("Model")
print(summary.to_string())

print("\n" + "="*65)
print("  SUMMARY – Fairness Metrics (Best Model: " + BEST + ")")
print("="*65)
best_fair = bias_df[bias_df["Model"] == BEST][
    ["Dimension","Subgroup","N","AUC","Sensitivity","Specificity","EOD","DPD"]
].round(3)
print(best_fair.to_string(index=False))

print("\n  All plots saved to:", OUTPUT)
print("  Script complete.\n")

"""
Titanic Survival Prediction - Mini Project (Data Science)
Problem : Develop a classification system to predict passenger survival using age, sex,
          passenger class, fare and other passenger information after handling missing values.
Dataset : Kaggle - Titanic Dataset (train.csv)
Run     : python titanic_survival.py
"""
import os, re, sys, json, urllib.request, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)

RANDOM_STATE = 42
CSV_PATH, OUT_DIR = "data/train.csv", "outputs"
BACKUP_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
os.makedirs("data", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# ------------------------------------------------------------- 1. LOAD DATA
if not os.path.exists(CSV_PATH):
    try:
        print("train.csv not found - downloading the Titanic dataset...")
        urllib.request.urlretrieve(BACKUP_URL, CSV_PATH)
    except Exception:
        sys.exit("Put the Kaggle Titanic file as data/train.csv and run again.")
df = pd.read_csv(CSV_PATH)
if "Survived" not in df.columns:
    sys.exit("This file has no 'Survived' column. Use Kaggle's train.csv.")
print("Shape:", df.shape)
print(df.head())
print("\nData types:\n", df.dtypes)

# ------------------------------------------------------- 2. MISSING VALUES
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Missing values": missing, "Percentage (%)": missing_pct})
miss_df = miss_df[miss_df["Missing values"] > 0].sort_values("Missing values", ascending=False)
miss_df.to_csv(f"{OUT_DIR}/missing_values.csv")
print("\nMissing values before handling:\n", miss_df)

plt.figure(figsize=(6.5, 3.8))
sns.barplot(x=miss_df.index, y=miss_df["Percentage (%)"], color="#1559a6")
plt.ylabel("Missing (%)"); plt.title("Missing Values Before Handling")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_missing_values.png", dpi=150); plt.close()

dups = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# --------------------------------------------- 3. FEATURE ENGINEERING + CLEANING
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
df["Has_Cabin"] = df["Cabin"].notna().astype(int)        # Cabin has too many gaps: keep only whether it is known
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

# Age: fill with the median age of passengers having the same Title and class
df["Age"] = df["Age"].fillna(df.groupby(["Title", "Pclass"])["Age"].transform("median"))
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
df["Age_Group"] = pd.cut(df["Age"], [0, 12, 18, 35, 60, 100], labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
print("\nMissing values after handling (used columns):",
      int(df[["Age", "Embarked", "Fare", "Title"]].isnull().sum().sum()))

# ---------------------------------------------------------- 4. VISUALISATION
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
sns.countplot(x="Survived", data=df, color="#1559a6", ax=ax[0]); ax[0].set_xticklabels(["Died", "Survived"])
ax[0].set_title("Survival Count")
sns.barplot(x="Sex", y="Survived", data=df, color="#1559a6", ax=ax[1], errorbar=None); ax[1].set_title("Survival Rate by Sex")
sns.barplot(x="Pclass", y="Survived", data=df, color="#1559a6", ax=ax[2], errorbar=None); ax[2].set_title("Survival Rate by Class")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/02_survival_overview.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.histplot(data=df, x="Age", hue="Survived", bins=25, multiple="stack", ax=ax[0], palette=["#c0392b", "#1559a6"])
ax[0].set_title("Age Distribution by Survival")
sns.barplot(x="Age_Group", y="Survived", data=df, color="#1559a6", ax=ax[1], errorbar=None)
ax[1].set_title("Survival Rate by Age Group"); ax[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_age_analysis.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
sns.boxplot(x="Pclass", y="Fare", data=df, color="#93c5fd", ax=ax[0]); ax[0].set_yscale("log"); ax[0].set_title("Fare by Class (log scale)")
sns.barplot(x="Embarked", y="Survived", data=df, color="#1559a6", ax=ax[1], errorbar=None); ax[1].set_title("Survival Rate by Port")
sns.barplot(x="Family_Size", y="Survived", data=df, color="#1559a6", ax=ax[2], errorbar=None); ax[2].set_title("Survival Rate by Family Size")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_fare_port_family.png", dpi=150); plt.close()

num_for_corr = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Has_Cabin", "Family_Size", "Is_Alone"]
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[num_for_corr].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8})
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_correlation_heatmap.png", dpi=150); plt.close()

# ---------------------------------------------------- 5. MODEL PREPARATION
num_cols = ["Age", "Fare", "SibSp", "Parch", "Family_Size"]
cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Has_Cabin", "Is_Alone"]
X = df[num_cols + cat_cols].copy()
X["Pclass"] = X["Pclass"].astype(str)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def make_pipe(model):
    pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    return Pipeline([("pre", pre), ("model", model)])


# ------------------------------------------------------ 6. TRAIN + COMPARE
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# small grid search to tune the Random Forest
grid = GridSearchCV(make_pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
                    {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, 8, None],
                     "model__min_samples_leaf": [1, 3, 5]}, cv=skf, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
print("\nBest Random Forest parameters:", best_params)
models["Random Forest (tuned)"] = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)

rows, fitted, probs = [], {}, {}
for name, m in models.items():
    pipe = make_pipe(m).fit(X_train, y_train)
    pred, prob = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
    fitted[name], probs[name] = pipe, prob
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                 "F1-Score": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, prob),
                 "5-Fold CV Acc": cross_val_score(make_pipe(m), X, y, cv=skf, scoring="accuracy").mean()})
res_df = pd.DataFrame(rows).round(4).sort_values("Accuracy", ascending=False)
res_df.to_csv(f"{OUT_DIR}/model_results.csv", index=False)
print("\n=== MODEL RESULTS ===\n", res_df.to_string(index=False))

best = res_df.iloc[0]["Model"]
best_pipe = fitted[best]
ConfusionMatrixDisplay(confusion_matrix(y_test, best_pipe.predict(X_test)),
                       display_labels=["Died", "Survived"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix ({best})"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
for name in res_df["Model"].head(4):
    fpr, tpr, _ = roc_curve(y_test, probs[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probs[name]):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves (top 4 models)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_roc_curves.png", dpi=150); plt.close()

# feature importance from the tuned Random Forest
rf_pipe = fitted["Random Forest (tuned)"]
names = rf_pipe.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=[n.split("__")[1] for n in names]).sort_values()
plt.figure(figsize=(7, 5.5)); imp.tail(12).plot(kind="barh", color="#1559a6")
plt.title("Top Feature Importances (Random Forest)"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_feature_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).round(4).to_csv(f"{OUT_DIR}/feature_importance.csv", header=["importance"])

# ----------------------------------------------- 7. PREDICT NEW PASSENGERS
sample = pd.DataFrame([
    dict(Age=28, Fare=80.0, SibSp=0, Parch=0, Family_Size=1, Pclass="1", Sex="female", Embarked="C", Title="Miss", Has_Cabin=1, Is_Alone=1),
    dict(Age=30, Fare=8.05, SibSp=0, Parch=0, Family_Size=1, Pclass="3", Sex="male", Embarked="S", Title="Mr", Has_Cabin=0, Is_Alone=1),
    dict(Age=6, Fare=30.0, SibSp=1, Parch=2, Family_Size=4, Pclass="2", Sex="male", Embarked="S", Title="Master", Has_Cabin=0, Is_Alone=0),
    dict(Age=45, Fare=15.0, SibSp=1, Parch=0, Family_Size=2, Pclass="3", Sex="female", Embarked="Q", Title="Mrs", Has_Cabin=0, Is_Alone=0)])
desc = ["28-year-old woman, 1st class", "30-year-old man, 3rd class, travelling alone",
        "6-year-old boy, 2nd class, with family", "45-year-old woman, 3rd class, with spouse"]
sp = best_pipe.predict_proba(sample[num_cols + cat_cols])[:, 1]
pred_tbl = pd.DataFrame({"Passenger": desc, "Survival probability": (sp * 100).round(1).astype(str) + " %",
                         "Prediction": np.where(sp >= 0.5, "Survived", "Died")})
pred_tbl.to_csv(f"{OUT_DIR}/sample_predictions.csv", index=False)
print("\n=== SAMPLE PREDICTIONS ===\n", pred_tbl.to_string(index=False))

# ----------------------------------------------- 8. FACTS FOR THE REPORT
rate = lambda col: {str(k): round(float(v) * 100, 1) for k, v in df.groupby(col)["Survived"].mean().items()}
facts = dict(n=len(df), duplicates=dups, survived=int(y.sum()), survival_rate=round(float(y.mean()) * 100, 1),
             by_sex=rate("Sex"), by_class=rate("Pclass"), by_port=rate("Embarked"), by_age_group=rate("Age_Group"),
             by_alone=rate("Is_Alone"), missing={k: [int(v["Missing values"]), float(v["Percentage (%)"])] for k, v in miss_df.iterrows()},
             best_model=best, best_params=best_params, top_features=list(imp.sort_values(ascending=False).index[:3]),
             n_train=len(X_train), n_test=len(X_test))
json.dump(facts, open(f"{OUT_DIR}/facts.json", "w"), indent=1)
print(f"\nBest model: {best}\nAll results saved in '{OUT_DIR}/'")

import pandas as pd
import numpy as np

df = pd.read_csv("QVI_data.csv")

df["DATE"] = pd.to_datetime(df["DATE"])
df["YEARMONTH"] = df["DATE"].dt.strftime("%Y%m").astype(int)


measure = (
    df.groupby(["STORE_NBR", "YEARMONTH"])
    .agg(
        totSales=("TOT_SALES", "sum"),
        nCustomers=("LYLTY_CARD_NBR", "nunique"),
        nTxn=("TXN_ID", "nunique"),
    )
    .reset_index()
)

measure["txnPerCust"] = measure["nTxn"] / measure["nCustomers"]


# PRE-TRIAL DATA


pre_trial = measure[measure["YEARMONTH"] < 201902].copy()


store_counts = pre_trial.groupby("STORE_NBR")["YEARMONTH"].nunique()

valid_stores = store_counts[store_counts == 12].index

pre_trial = pre_trial.copy()


# CORRELATION FUNCTION


def calc_corr(input_df, metric, trial_store):

    results = []

    trial = input_df[input_df["STORE_NBR"] == trial_store][["YEARMONTH", metric]]

    for store in input_df["STORE_NBR"].unique():

        if store == trial_store:
            continue

        comp = input_df[input_df["STORE_NBR"] == store][["YEARMONTH", metric]]

        merged = trial.merge(comp, on="YEARMONTH", suffixes=("_t", "_c"))

        merged = merged.dropna()

        if len(merged) < 6:
            continue

        corr = merged[f"{metric}_t"].corr(merged[f"{metric}_c"])

        if pd.isna(corr):
            continue

        results.append([trial_store, store, corr])

    return pd.DataFrame(results, columns=["Store1", "Store2", "corr"])


# MAGNITUDE FUNCTION


def calc_magnitude(input_df, metric, trial_store):

    results = []

    trial = input_df[input_df["STORE_NBR"] == trial_store][["YEARMONTH", metric]]

    for store in input_df["STORE_NBR"].unique():

        if store == trial_store:
            continue

        comp = input_df[input_df["STORE_NBR"] == store][["YEARMONTH", metric]]

        merged = trial.merge(comp, on="YEARMONTH", suffixes=("_t", "_c"))

        if merged.shape[0] < 3:
            continue

        diff = abs(merged[f"{metric}_t"] - merged[f"{metric}_c"])

        if diff.max() == diff.min():
            mag = 1.0
        else:
            mag = 1 - (diff - diff.min()) / (diff.max() - diff.min())

        results.append([trial_store, store, mag.mean()])

    return pd.DataFrame(results, columns=["Store1", "Store2", "mag"])


# FIND CONTROL STORE


def find_control(trial_store):

    corr_sales = calc_corr(pre_trial, "totSales", trial_store)
    corr_cust = calc_corr(pre_trial, "nCustomers", trial_store)

    mag_sales = calc_magnitude(pre_trial, "totSales", trial_store)
    mag_cust = calc_magnitude(pre_trial, "nCustomers", trial_store)

    if corr_sales.empty or mag_sales.empty:
        raise ValueError(f"No candidates for trial store {trial_store}")

    corr_w = 0.5

    score_sales = corr_sales.merge(mag_sales, on=["Store1", "Store2"])
    score_sales["score_sales"] = (
        corr_w * score_sales["corr"] + (1 - corr_w) * score_sales["mag"]
    )

    score_cust = corr_cust.merge(mag_cust, on=["Store1", "Store2"])
    score_cust["score_cust"] = (
        corr_w * score_cust["corr"] + (1 - corr_w) * score_cust["mag"]
    )

    score = score_sales.merge(score_cust, on=["Store1", "Store2"])

    score["final"] = (score["score_sales"] + score["score_cust"]) / 2

    score = score.dropna(subset=["final"])

    score = score[score["Store2"] != trial_store]

    if score.empty:
        raise ValueError(f"No valid control store for {trial_store}")

    control = score.sort_values("final", ascending=False).iloc[0]["Store2"]

    return int(control)


# TRIAL EVALUATION


def evaluate_trial(trial_store, control_store):

    pre = measure[measure["YEARMONTH"] < 201902]

    trial_pre = pre[pre["STORE_NBR"] == trial_store]["totSales"].sum()
    control_pre = pre[pre["STORE_NBR"] == control_store]["totSales"].sum()

    scaling = trial_pre / control_pre

    temp = measure.copy()

    temp.loc[temp["STORE_NBR"] == control_store, "scaledControl"] = (
        temp["totSales"] * scaling
    )

    trial_df = temp[temp["STORE_NBR"] == trial_store][["YEARMONTH", "totSales"]]
    control_df = temp[temp["STORE_NBR"] == control_store][
        ["YEARMONTH", "scaledControl"]
    ]

    comp = trial_df.merge(control_df, on="YEARMONTH")

    comp["pctDiff"] = (comp["totSales"] - comp["scaledControl"]) / comp["scaledControl"]

    std = comp[comp["YEARMONTH"] < 201902]["pctDiff"].std()

    comp["tValue"] = comp["pctDiff"] / std

    return comp, std


# RUN ALL TRIAL STORES

trial_stores = [77, 86, 88]

results = {}

for t in trial_stores:

    print("\n==============================")
    print("TRIAL STORE:", t)

    control = find_control(t)
    print("CONTROL STORE:", control)

    comp, std = evaluate_trial(t, control)

    print("STD:", std)
    print(comp.tail())

    results[t] = {"control": control, "comparison": comp}


#  SUMMARY


print("\n FINAL RESULT")

for t in trial_stores:
    print(f"Trial {t} → Control {results[t]['control']}")


import matplotlib.pyplot as plt

# =========================================================
# CREATE PLOTS (TRIAL VS CONTROL) + EXPORT PNG
# =========================================================


def plot_trial_vs_control(trial_store, control_store, metric="totSales"):

    data_plot = measure.copy()

    # label stores
    data_plot["StoreType"] = "Other"
    data_plot.loc[data_plot["STORE_NBR"] == trial_store, "StoreType"] = "Trial"
    data_plot.loc[data_plot["STORE_NBR"] == control_store, "StoreType"] = "Control"

    # aggregate monthly
    agg = data_plot.groupby(["YEARMONTH", "StoreType"])[metric].mean().reset_index()

    pivot = agg.pivot(index="YEARMONTH", columns="StoreType", values=metric)

    plt.figure(figsize=(10, 5))

    for col in pivot.columns:
        plt.plot(pivot.index.astype(str), pivot[col], label=col)

    plt.axvspan("201902", "201904", color="grey", alpha=0.2, label="Trial Period")

    plt.title(f"{metric} – Trial vs Control (Store {trial_store})")
    plt.xlabel("Month")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend()

    filename = f"plot_{metric}_trial_{trial_store}_vs_{control_store}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")


# =========================================================
# RUN PLOTS FOR ALL TRIAL STORES
# =========================================================

for t in trial_stores:
    c = results[t]["control"]

    plot_trial_vs_control(t, c, "totSales")
    plot_trial_vs_control(t, c, "nCustomers")

import matplotlib.pyplot as plt

# COMBINED TRIAL vs CONTROL vs OTHER STORES


def combined_trial_plot(trial_store, control_store):

    temp = measure.copy()

    # Label stores
    temp["StoreType"] = "Other Stores"

    temp.loc[temp["STORE_NBR"] == trial_store, "StoreType"] = (
        f"Trial Store {trial_store}"
    )

    temp.loc[temp["STORE_NBR"] == control_store, "StoreType"] = (
        f"Control Store {control_store}"
    )

    # Average other stores
    monthly = temp.groupby(["YEARMONTH", "StoreType"])["totSales"].mean().reset_index()

    # Normalize for easier comparison
    monthly["scaledSales"] = monthly.groupby("StoreType")["totSales"].transform(
        lambda x: x / x.iloc[0] * 100
    )

    # Plot
    plt.figure(figsize=(12, 6))

    for store in monthly["StoreType"].unique():

        subset = monthly[monthly["StoreType"] == store]

        plt.plot(
            subset["YEARMONTH"].astype(str),
            subset["scaledSales"],
            label=store,
            linewidth=3 if "Trial" in store else 2,
        )

    # Highlight trial period
    plt.axvspan("201902", "201904", alpha=0.2)

    plt.title(f"Trial vs Control vs Other Stores – Store {trial_store}")

    plt.xlabel("Month")
    plt.ylabel("Indexed Sales (Base = 100)")
    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()

    filename = f"combined_trial_{trial_store}.png"

    plt.savefig(filename, dpi=300)

    plt.close()

    print(f"Saved: {filename}")


# =========================================================
# CREATE FOR ALL TRIAL STORES
# =========================================================

combined_trial_plot(77, 233)
combined_trial_plot(86, 155)
combined_trial_plot(88, 178)


"""


FINAL REVIEW – INITIAL FINDINGS 


1. CONTROL STORE SELECTION

The control stores were selected based on a combined similarity score
that measures correlation and magnitude differences in both total sales
and number of customers during the pre-trial period.

Final selected control stores:
- Trial Store 77 → Control Store 233
- Trial Store 86 → Control Store 155
- Trial Store 88 → Control Store 178

These control stores were identified as the most comparable in terms of
historical sales and customer behaviour patterns prior to the trial period.

---------------------------------------------------------

2. TRIAL PERFORMANCE OVERVIEW

Trial stores 77 and 88 show noticeable differences in performance compared
to their respective control stores during the trial period.

- Store 77:
  Sales performance is significantly higher than the control store during
  the trial period, suggesting a positive impact from the trial layout.

- Store 86:
  While customer numbers increased during the trial period, total sales
  did not show a statistically significant uplift compared to the control store.
  This may indicate a change in purchasing behaviour (more customers, but
  lower average spend per customer).

- Store 88:
  Shows a positive difference in performance compared to its control store,
  indicating a potential uplift in sales and customer engagement during the trial.

---------------------------------------------------------

3. KEY INSIGHTS

- The trial layout appears to have had a positive impact on stores 77 and 88,
  with increased sales relative to their control stores.
- Store 86 shows increased customer engagement but no corresponding increase
  in sales, suggesting potential pricing or product mix effects.
- Results indicate that the trial is generally successful, but the impact
  is not uniform across all stores.

---------------------------------------------------------

4. CONCLUSION

Overall, the trial results suggest a positive impact of the new layout in
selected stores. However, further investigation is recommended for Store 86
to understand why increased customer numbers did not translate into higher sales.

As this is an early-stage analysis, these findings should be considered
preliminary. Further statistical testing and longer-term performance tracking
are recommended before making a full rollout decision.

=========================================================
"""

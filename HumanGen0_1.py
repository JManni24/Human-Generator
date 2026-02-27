import re
import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds

MENTAL = [
    "Empathy", "Communication", "Patience", "Discipline", "Leadership",
    "Adaptability", "Creativity", "Focus", "Logic", "Wisdom"
]
BIO = ["Height", "Intellect", "LifeExp", "Strength", "Weight"]

GLOBAL_MIN = {"Weight": 20.0, "Height": 30.0, "LifeExp": 10.0}
DEFAULT_RACK_CAPACITY = 100.0


def tier_num(t):
    if isinstance(t, str):
        m = re.search(r"(\d+)", t)
        return int(m.group(1)) if m else 0
    if pd.isna(t):
        return 0
    return int(t)


def normalize_text(s):
    if isinstance(s, str):
        return s.replace("\u00a0", " ").strip()
    return s


def load_tables_from_excel(file_obj):
    careers = pd.read_excel(file_obj, sheet_name="Careers")
    food = pd.read_excel(file_obj, sheet_name="Food")
    mem = pd.read_excel(file_obj, sheet_name="Memories")

    careers = careers.rename(columns={
        "Teir": "Tier",
        "Intelect": "Intellect",
        "Communications": "Communication",
        "Disipline": "Discipline",
        "Life Exp.": "LifeExp",
    })

    food = food.rename(columns={
        "Unnamed: 0": "Food",
        "Food Weight (Kg)": "ItemWeight",
        "Life Exp.": "LifeExp",
    })

    mem = mem.rename(columns={
        "Unnamed: 0": "Memory",
        "Memory Weight (Kg)": "ItemWeight",
        "Disipline": "Discipline",
    })

    if "Food" not in food.columns:
        raise ValueError("Food sheet must have item names in the first column (expected 'Unnamed: 0' or 'Food').")
    if "Memory" not in mem.columns:
        raise ValueError("Memories sheet must have item names in the first column (expected 'Unnamed: 0' or 'Memory').")

    food["Food"] = food["Food"].apply(normalize_text)
    mem["Memory"] = mem["Memory"].apply(normalize_text)

    if "Career" not in careers.columns or "Job" not in careers.columns:
        raise ValueError("Careers sheet must include 'Career' and 'Job' columns.")

    careers["TierNum"] = careers["Tier"].apply(tier_num)

    for col in BIO:
        if col in food.columns:
            food[col] = pd.to_numeric(food[col], errors="coerce").fillna(0)
        else:
            food[col] = 0

    for col in MENTAL:
        if col in mem.columns:
            mem[col] = pd.to_numeric(mem[col], errors="coerce").fillna(0)
        else:
            mem[col] = 0

    for col in BIO + MENTAL:
        if col in careers.columns:
            careers[col] = pd.to_numeric(careers[col], errors="coerce")
        else:
            careers[col] = np.nan

    if "ItemWeight" not in food.columns or "ItemWeight" not in mem.columns:
        raise ValueError("Food and Memories sheets must include 'Food Weight (Kg)' and 'Memory Weight (Kg)'.")

    food["ItemWeight"] = pd.to_numeric(food["ItemWeight"], errors="coerce").fillna(0)
    mem["ItemWeight"] = pd.to_numeric(mem["ItemWeight"], errors="coerce").fillna(0)

    if "FoodTier" not in food.columns:
        raise ValueError("Food sheet must include a 'FoodTier' column with values like T1, T2, T3, T4.")

    food["FoodTier"] = food["FoodTier"].astype(str).str.strip().str.upper()
    food.loc[~food["FoodTier"].isin(["T1", "T2", "T3", "T4"]), "FoodTier"] = "T1"

    return careers, food, mem


def tier_cost_map(t1, t2, t3, t4):
    return {"T1": float(t1), "T2": float(t2), "T3": float(t3), "T4": float(t4)}


def solve_global_memory_limited_food_unlimited(
    careers, food, mem,
    inv_mem,
    seeds_to_use,
    capacity_kg,
    enforce_diversity=False,
    memory_objective="count",
    tier_weights=True,
    food_tier_costs=None
):
    jobs = careers.reset_index(drop=True)
    J = len(jobs)

    food_names = food["Food"].tolist()
    mem_names = mem["Memory"].tolist()
    nF, nM = len(food_names), len(mem_names)

    invM = np.array([inv_mem.get(n, 0) for n in mem_names], dtype=float)

    wF = food["ItemWeight"].to_numpy()
    wM = mem["ItemWeight"].to_numpy()

    # Big-M upper bounds for food variables so MILP stays bounded.
    # Each human has at most capacity_kg of ingredient weight, so total food items per human is limited.
    min_food_w = float(np.min(wF[wF > 0])) if np.any(wF > 0) else 1.0
    max_food_items_per_human = int(np.floor(float(capacity_kg) / min_food_w)) if min_food_w > 0 else 100
    if max_food_items_per_human < 1:
        max_food_items_per_human = 1
    ub_food_per_job_item = int(seeds_to_use) * max_food_items_per_human

    idx_y_start = 0
    idx_xF_start = idx_y_start + J
    idx_xM_start = idx_xF_start + (J * nF)
    n_vars = J + (J * nF) + (J * nM)

    def y_idx(j): return idx_y_start + j
    def xF_idx(j, i): return idx_xF_start + j * nF + i
    def xM_idx(j, k): return idx_xM_start + j * nM + k

    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)

    # y_j, number of humans of job j
    for j in range(J):
        ub[y_idx(j)] = float(seeds_to_use)

    # xF_j_i, total food items i used across all humans of job j
    for j in range(J):
        for i in range(nF):
            ub[xF_idx(j, i)] = float(ub_food_per_job_item)

    # xM_j_k, total memory items k used across all humans of job j, bounded by inventory
    for j in range(J):
        for k in range(nM):
            ub[xM_idx(j, k)] = float(invM[k])

    integrality = np.ones(n_vars, dtype=int)
    constraints = []

    # total humans <= seeds_to_use
    A = np.zeros((1, n_vars))
    for j in range(J):
        A[0, y_idx(j)] = 1
    constraints.append(LinearConstraint(A, 0, float(seeds_to_use)))

    # total memory usage <= inventory per memory item
    for k in range(nM):
        A = np.zeros((1, n_vars))
        for j in range(J):
            A[0, xM_idx(j, k)] = 1
        constraints.append(LinearConstraint(A, 0, float(invM[k])))

    # per job rack capacity, sum item weights <= capacity * humans
    for j in range(J):
        A = np.zeros((1, n_vars))
        A[0, y_idx(j)] = -float(capacity_kg)
        for i in range(nF):
            A[0, xF_idx(j, i)] = float(wF[i])
        for k in range(nM):
            A[0, xM_idx(j, k)] = float(wM[k])
        constraints.append(LinearConstraint(A, -np.inf, 0.0))

    # per job requirements, traits >= req * humans
    for j in range(J):
        row = jobs.loc[j]

        # BIO, from food only
        for tr in BIO:
            req = row.get(tr, np.nan)
            if pd.isna(req):
                req = 0.0
            if tr in GLOBAL_MIN:
                req = max(float(req), float(GLOBAL_MIN[tr]))

            A = np.zeros((1, n_vars))
            A[0, y_idx(j)] = -float(req)
            for i in range(nF):
                A[0, xF_idx(j, i)] = float(food.loc[i, tr])
            constraints.append(LinearConstraint(A, 0.0, np.inf))

        # MENTAL, from memories only
        for tr in MENTAL:
            req = row.get(tr, np.nan)
            if pd.isna(req):
                continue
            A = np.zeros((1, n_vars))
            A[0, y_idx(j)] = -float(req)
            for k in range(nM):
                A[0, xM_idx(j, k)] = float(mem.loc[k, tr])
            constraints.append(LinearConstraint(A, 0.0, np.inf))

    # diversity, at least one top tier human for each Career category
    if enforce_diversity:
        for cg in sorted(jobs["Career"].dropna().unique().tolist()):
            subset = jobs[jobs["Career"] == cg]
            if subset.empty:
                continue
            top_tier = int(subset["TierNum"].max())
            top_jobs_idx = subset[subset["TierNum"] == top_tier].index.tolist()
            A = np.zeros((1, n_vars))
            for j in top_jobs_idx:
                A[0, y_idx(int(j))] = 1
            constraints.append(LinearConstraint(A, 1.0, np.inf))

    # objective
    c = np.zeros(n_vars)

    # Primary goal, maximize tier quality
    # Secondary goal, maximize number of humans
    # Tertiary goal, minimize memory consumption
    # Quaternary goal, minimize food rarity cost
    # MILP is minimization, so maximize becomes negative coefficients.

    if tier_weights:
        tier_big = 100000.0
        count_mid = 1000.0
    else:
        tier_big = 0.0
        count_mid = 1000.0

    for j in range(J):
        c[y_idx(j)] = -(tier_big * float(jobs.loc[j, "TierNum"]) + count_mid * 1.0)

    # memory cost
    if memory_objective == "weight":
        mem_cost = wM
    else:
        mem_cost = np.ones(nM)

    mem_small = 1.0
    for j in range(J):
        for k in range(nM):
            c[xM_idx(j, k)] = float(mem_small * mem_cost[k])

    # food tier cost, unlimited food but minimize painful ingredients
    if food_tier_costs is None:
        food_tier_costs = {"T1": 1.0, "T2": 3.0, "T3": 9.0, "T4": 27.0}

    food_cost_vec = np.array([food_tier_costs.get(t, 1.0) for t in food["FoodTier"].tolist()], dtype=float)
    food_tiny = 0.01
    for j in range(J):
        for i in range(nF):
            c[xF_idx(j, i)] = float(food_tiny * food_cost_vec[i])

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints
    )

    if res.status != 0:
        return None, f"Global solver infeasible or failed, status {res.status}. Try turning off diversity, lowering seeds, or increasing memory inventory."

    x = res.x
    y = np.array([int(round(x[y_idx(j)])) for j in range(J)], dtype=int)
    chosen_jobs = np.where(y > 0)[0].tolist()

    plan_rows = []
    for j in chosen_jobs:
        row = jobs.loc[j]
        yj = int(y[j])

        used_mem = {}
        for k in range(nM):
            val = int(round(x[xM_idx(j, k)]))
            if val > 0:
                used_mem[mem_names[k]] = val

        used_food = {}
        for i in range(nF):
            val = int(round(x[xF_idx(j, i)]))
            if val > 0:
                used_food[food_names[i]] = val

        plan_rows.append({
            "Career": row["Career"],
            "Job": row["Job"],
            "Tier": row["Tier"],
            "TierNum": int(row["TierNum"]),
            "Humans": yj,
            "MemUsed": used_mem,
            "FoodUsed": used_food
        })

    # aggregates
    total_humans = int(y.sum())

    tier_counts = {}
    for r in plan_rows:
        tier_counts[r["TierNum"]] = tier_counts.get(r["TierNum"], 0) + int(r["Humans"])

    mem_agg = {}
    for r in plan_rows:
        for k, v in r["MemUsed"].items():
            mem_agg[k] = mem_agg.get(k, 0) + int(v)

    food_agg = {}
    for r in plan_rows:
        for k, v in r["FoodUsed"].items():
            food_agg[k] = food_agg.get(k, 0) + int(v)

    # food tier breakdown
    food_lookup_tier = dict(zip(food["Food"].tolist(), food["FoodTier"].tolist()))
    food_by_tier = {"T1": {}, "T2": {}, "T3": {}, "T4": {}}
    food_tier_totals = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    for fname, cnt in food_agg.items():
        t = food_lookup_tier.get(fname, "T1")
        food_by_tier.setdefault(t, {})
        food_by_tier[t][fname] = cnt
        food_tier_totals[t] = food_tier_totals.get(t, 0) + int(cnt)

    summary = {
        "SeedsRequested": int(seeds_to_use),
        "TotalHumansPlanned": int(total_humans),
        "TierCounts": dict(sorted(tier_counts.items(), key=lambda x: -x[0])),
        "TotalMemoryItemsUsed": int(sum(mem_agg.values())),
        "TotalFoodItemsRequired": int(sum(food_agg.values())),
        "FoodItemsByTierTotals": food_tier_totals
    }

    return {
        "summary": summary,
        "plan_rows": sorted(plan_rows, key=lambda r: (-r["TierNum"], r["Career"], r["Job"])),
        "memory_required": mem_agg,
        "food_required": food_agg,
        "food_required_by_tier": food_by_tier
    }, None


def df_from_dict(d, k1, k2):
    if not d:
        return pd.DataFrame(columns=[k1, k2])
    return pd.DataFrame(sorted(d.items(), key=lambda x: (-x[1], x[0])), columns=[k1, k2])


st.set_page_config(page_title="Last Caretaker Planner, Memory Limited", layout="wide")
st.title("Last Caretaker Planner, Memory Limited")

uploaded = st.file_uploader("Upload Human Calculator.xlsx", type=["xlsx"])
if uploaded is None:
    st.info("Upload your workbook to begin.")
    st.stop()

try:
    careers_df, food_df, mem_df = load_tables_from_excel(uploaded)
except Exception as e:
    st.error(f"Failed to load workbook: {e}")
    st.stop()

st.subheader("Planner settings")
c1, c2, c3, c4 = st.columns(4)
with c1:
    capacity = st.number_input("Rack ingredient capacity, Kg", min_value=1.0, value=float(DEFAULT_RACK_CAPACITY), step=1.0)
with c2:
    seeds_to_use = st.number_input("Seeds to plan for", min_value=1, value=85, step=1)
with c3:
    memory_objective = st.selectbox("Memory economy rule", options=["count", "weight"], index=0)
with c4:
    enforce_div = st.checkbox("Diversity, at least one top tier per Career category", value=False)

st.markdown("Food rarity costs, lower means the solver prefers it")
t1c, t2c, t3c, t4c = st.columns(4)
with t1c:
    cost_t1 = st.number_input("T1 cost", min_value=0.0, value=1.0, step=0.5)
with t2c:
    cost_t2 = st.number_input("T2 cost", min_value=0.0, value=3.0, step=0.5)
with t3c:
    cost_t3 = st.number_input("T3 cost", min_value=0.0, value=9.0, step=0.5)
with t4c:
    cost_t4 = st.number_input("T4 cost", min_value=0.0, value=27.0, step=0.5)

st.subheader("Memory inventory")
inv_mem = {}
left, right = st.columns(2)
mem_names = mem_df["Memory"].tolist()
half = int(np.ceil(len(mem_names) / 2))

with left:
    for name in mem_names[:half]:
        inv_mem[name] = st.number_input(name, min_value=0, value=0, step=1, key=f"mem_{name}_l")
with right:
    for name in mem_names[half:]:
        inv_mem[name] = st.number_input(name, min_value=0, value=0, step=1, key=f"mem_{name}_r")

st.subheader("Run global planner, memory limited, food unlimited")
if st.button("Compute plan and required food"):
    with st.spinner("Solving globally..."):
        result, err = solve_global_memory_limited_food_unlimited(
            careers_df,
            food_df,
            mem_df,
            inv_mem=inv_mem,
            seeds_to_use=int(seeds_to_use),
            capacity_kg=float(capacity),
            enforce_diversity=enforce_div,
            memory_objective=memory_objective,
            tier_weights=True,
            food_tier_costs=tier_cost_map(cost_t1, cost_t2, cost_t3, cost_t4)
        )

    if err:
        st.warning(err)
        st.stop()

    st.success("Plan solved.")

    st.markdown("Summary")
    st.write(result["summary"])

    st.markdown("How many humans you can make at each tier")
    tier_counts = result["summary"]["TierCounts"]
    tier_df = pd.DataFrame([{"TierNum": k, "Humans": v} for k, v in tier_counts.items()]).sort_values("TierNum", ascending=False)
    st.dataframe(tier_df, use_container_width=True)

    st.markdown("Humans by job")
    job_df = pd.DataFrame([{
        "TierNum": r["TierNum"],
        "Tier": r["Tier"],
        "Career": r["Career"],
        "Job": r["Job"],
        "Humans": r["Humans"]
    } for r in result["plan_rows"]]).sort_values(["TierNum", "Humans"], ascending=[False, False])
    st.dataframe(job_df, use_container_width=True)

    st.markdown("Required Memory items")
    st.dataframe(df_from_dict(result["memory_required"], "Memory", "Count"), use_container_width=True)

    st.markdown("Required Food items")
    st.dataframe(df_from_dict(result["food_required"], "Food", "Count"), use_container_width=True)

    st.markdown("Required Food, grouped by FoodTier")
    ft = result["food_required_by_tier"]
    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.markdown("T1")
        st.dataframe(df_from_dict(ft.get("T1", {}), "Food", "Count"), use_container_width=True)
    with cB:
        st.markdown("T2")
        st.dataframe(df_from_dict(ft.get("T2", {}), "Food", "Count"), use_container_width=True)
    with cC:
        st.markdown("T3")
        st.dataframe(df_from_dict(ft.get("T3", {}), "Food", "Count"), use_container_width=True)
    with cD:
        st.markdown("T4")
        st.dataframe(df_from_dict(ft.get("T4", {}), "Food", "Count"), use_container_width=True)

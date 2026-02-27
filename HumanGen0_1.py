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

VALID_FOOD_TIERS = ["T1", "T2", "T3", "T4"]
VALID_MEM_TIERS = ["T0", "T1", "T2", "T3", "T4"]


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


def normalize_tier(x, valid, default):
    if pd.isna(x):
        return default
    s = str(x).strip().upper()
    return s if s in valid else default


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
        raise ValueError("Food sheet must include a 'FoodTier' column with values T1, T2, T3, T4.")
    food["FoodTier"] = food["FoodTier"].apply(lambda x: normalize_tier(x, VALID_FOOD_TIERS, "T1"))

    if "MemoryTier" not in mem.columns:
        raise ValueError("Memories sheet must include a 'MemoryTier' column with values T0, T1, T2, T3, T4.")
    mem["MemoryTier"] = mem["MemoryTier"].apply(lambda x: normalize_tier(x, VALID_MEM_TIERS, "T1"))

    return careers, food, mem


def costs_from_inputs(food_costs, mem_costs):
    food_map = {k: float(v) for k, v in food_costs.items()}
    mem_map = {k: float(v) for k, v in mem_costs.items()}
    return food_map, mem_map


def df_from_dict(d, k1, k2):
    if not d:
        return pd.DataFrame(columns=[k1, k2])
    return pd.DataFrame(sorted(d.items(), key=lambda x: (-x[1], x[0])), columns=[k1, k2])


def aggregate_campaign_solver(
    careers, food, mem,
    inv_mem,
    seeds_to_use,
    capacity_kg,
    enforce_diversity=False,
    memory_objective="count",
    food_tier_costs=None,
    mem_tier_costs=None
):
    jobs = careers.reset_index(drop=True)
    J = len(jobs)

    food_names = food["Food"].tolist()
    mem_names = mem["Memory"].tolist()
    nF, nM = len(food_names), len(mem_names)

    invM = np.array([inv_mem.get(n, 0) for n in mem_names], dtype=float)

    wF = food["ItemWeight"].to_numpy()
    wM = mem["ItemWeight"].to_numpy()

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

    for j in range(J):
        ub[y_idx(j)] = float(seeds_to_use)

    for j in range(J):
        for i in range(nF):
            ub[xF_idx(j, i)] = float(ub_food_per_job_item)

    for j in range(J):
        for k in range(nM):
            ub[xM_idx(j, k)] = float(invM[k])

    integrality = np.ones(n_vars, dtype=int)
    constraints = []

    A = np.zeros((1, n_vars))
    for j in range(J):
        A[0, y_idx(j)] = 1
    constraints.append(LinearConstraint(A, 0, float(seeds_to_use)))

    for k in range(nM):
        A = np.zeros((1, n_vars))
        for j in range(J):
            A[0, xM_idx(j, k)] = 1
        constraints.append(LinearConstraint(A, 0, float(invM[k])))

    for j in range(J):
        A = np.zeros((1, n_vars))
        A[0, y_idx(j)] = -float(capacity_kg)
        for i in range(nF):
            A[0, xF_idx(j, i)] = float(wF[i])
        for k in range(nM):
            A[0, xM_idx(j, k)] = float(wM[k])
        constraints.append(LinearConstraint(A, -np.inf, 0.0))

    for j in range(J):
        row = jobs.loc[j]

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

        for tr in MENTAL:
            req = row.get(tr, np.nan)
            if pd.isna(req):
                continue
            A = np.zeros((1, n_vars))
            A[0, y_idx(j)] = -float(req)
            for k in range(nM):
                A[0, xM_idx(j, k)] = float(mem.loc[k, tr])
            constraints.append(LinearConstraint(A, 0.0, np.inf))

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

    c = np.zeros(n_vars)

    tier_big = 100000.0
    count_mid = 1000.0

    for j in range(J):
        c[y_idx(j)] = -(tier_big * float(jobs.loc[j, "TierNum"]) + count_mid)

    if mem_tier_costs is None:
        mem_tier_costs = {"T0": 0.5, "T1": 1.0, "T2": 3.0, "T3": 9.0, "T4": 27.0}
    mem_cost_vec = np.array([mem_tier_costs.get(t, 1.0) for t in mem["MemoryTier"].tolist()], dtype=float)

    if memory_objective == "weight":
        mem_cost_vec = mem_cost_vec * wM
    else:
        mem_cost_vec = mem_cost_vec * 1.0

    mem_small = 1.0
    for j in range(J):
        for k in range(nM):
            c[xM_idx(j, k)] = float(mem_small * mem_cost_vec[k])

    if food_tier_costs is None:
        food_tier_costs = {"T1": 1.0, "T2": 3.0, "T3": 9.0, "T4": 27.0}
    food_cost_vec = np.array([food_tier_costs.get(t, 1.0) for t in food["FoodTier"].tolist()], dtype=float)

    food_tiny = 0.01
    for j in range(J):
        for i in range(nF):
            c[xF_idx(j, i)] = float(food_tiny * food_cost_vec[i])

    res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints)

    if res.status != 0:
        return None, f"Campaign solver infeasible or failed, status {res.status}. Try turning off diversity or increasing memory inventory."

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

    mem_agg = {}
    for r in plan_rows:
        for k, v in r["MemUsed"].items():
            mem_agg[k] = mem_agg.get(k, 0) + int(v)

    food_agg = {}
    for r in plan_rows:
        for k, v in r["FoodUsed"].items():
            food_agg[k] = food_agg.get(k, 0) + int(v)

    tier_counts = {}
    for r in plan_rows:
        tier_counts[r["TierNum"]] = tier_counts.get(r["TierNum"], 0) + int(r["Humans"])

    summary = {
        "SeedsRequested": int(seeds_to_use),
        "TotalHumansPlanned": int(sum(tier_counts.values())),
        "TierCounts": dict(sorted(tier_counts.items(), key=lambda z: -z[0])),
        "TotalMemoryItemsUsed": int(sum(mem_agg.values())),
        "TotalFoodItemsRequired": int(sum(food_agg.values()))
    }

    return {
        "summary": summary,
        "plan_rows": sorted(plan_rows, key=lambda r: (-r["TierNum"], r["Career"], r["Job"])),
        "memory_required": mem_agg,
        "food_required": food_agg
    }, None


def small_batch_slot_solver(
    careers, food, mem,
    inv_mem,
    seeds_to_use,
    capacity_kg,
    unique_careers=True,
    unique_jobs=False,
    maximize_career_diversity=True,
    memory_objective="count",
    food_tier_costs=None,
    mem_tier_costs=None
):
    jobs = careers.reset_index(drop=True)
    J = len(jobs)
    S = int(seeds_to_use)

    food_names = food["Food"].tolist()
    mem_names = mem["Memory"].tolist()
    nF, nM = len(food_names), len(mem_names)

    invM = np.array([inv_mem.get(n, 0) for n in mem_names], dtype=float)

    wF = food["ItemWeight"].to_numpy()
    wM = mem["ItemWeight"].to_numpy()

    # Upper bound for food per slot
    min_food_w = float(np.min(wF[wF > 0])) if np.any(wF > 0) else 1.0
    max_food_items_per_human = int(np.floor(float(capacity_kg) / min_food_w)) if min_food_w > 0 else 100
    if max_food_items_per_human < 1:
        max_food_items_per_human = 1

    # Variable layout
    # y[s,j] binary choose job j for slot s
    # xF[s,i] integer food counts in slot s
    # xM[s,k] integer memory counts in slot s
    idx_y_start = 0
    idx_xF_start = idx_y_start + (S * J)
    idx_xM_start = idx_xF_start + (S * nF)
    n_vars = (S * J) + (S * nF) + (S * nM)

    def y_idx(s, j): return idx_y_start + s * J + j
    def xF_idx(s, i): return idx_xF_start + s * nF + i
    def xM_idx(s, k): return idx_xM_start + s * nM + k

    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)

    # y binary
    for s in range(S):
        for j in range(J):
            ub[y_idx(s, j)] = 1.0

    # food per slot bounded
    for s in range(S):
        for i in range(nF):
            ub[xF_idx(s, i)] = float(max_food_items_per_human)

    # memory per slot bounded by inventory (loose but fine)
    for s in range(S):
        for k in range(nM):
            ub[xM_idx(s, k)] = float(invM[k])

    integrality = np.ones(n_vars, dtype=int)
    constraints = []

    # Each slot chooses exactly one job
    for s in range(S):
        A = np.zeros((1, n_vars))
        for j in range(J):
            A[0, y_idx(s, j)] = 1.0
        constraints.append(LinearConstraint(A, 1.0, 1.0))

    # Total memory across slots <= inventory
    for k in range(nM):
        A = np.zeros((1, n_vars))
        for s in range(S):
            A[0, xM_idx(s, k)] = 1.0
        constraints.append(LinearConstraint(A, 0.0, float(invM[k])))

    # Rack capacity per slot
    for s in range(S):
        A = np.zeros((1, n_vars))
        for i in range(nF):
            A[0, xF_idx(s, i)] = float(wF[i])
        for k in range(nM):
            A[0, xM_idx(s, k)] = float(wM[k])
        constraints.append(LinearConstraint(A, -np.inf, float(capacity_kg)))

    # Link xF and xM to chosen job using big M
    # For each slot and each job, enforce requirements only if y=1
    # For bio: sum food_trait >= req * y
    for s in range(S):
        for j in range(J):
            row = jobs.loc[j]

            for tr in BIO:
                req = row.get(tr, np.nan)
                if pd.isna(req):
                    req = 0.0
                if tr in GLOBAL_MIN:
                    req = max(float(req), float(GLOBAL_MIN[tr]))

                A = np.zeros((1, n_vars))
                for i in range(nF):
                    A[0, xF_idx(s, i)] = float(food.loc[i, tr])
                A[0, y_idx(s, j)] = -float(req)
                constraints.append(LinearConstraint(A, 0.0, np.inf))

            for tr in MENTAL:
                req = row.get(tr, np.nan)
                if pd.isna(req):
                    continue
                A = np.zeros((1, n_vars))
                for k in range(nM):
                    A[0, xM_idx(s, k)] = float(mem.loc[k, tr])
                A[0, y_idx(s, j)] = -float(req)
                constraints.append(LinearConstraint(A, 0.0, np.inf))

    # Uniqueness constraints
    if unique_jobs:
        for j in range(J):
            A = np.zeros((1, n_vars))
            for s in range(S):
                A[0, y_idx(s, j)] = 1.0
            constraints.append(LinearConstraint(A, 0.0, 1.0))

    if unique_careers:
        careers_list = sorted(jobs["Career"].dropna().unique().tolist())
        for cg in careers_list:
            idxs = jobs.index[jobs["Career"] == cg].tolist()
            A = np.zeros((1, n_vars))
            for s in range(S):
                for j in idxs:
                    A[0, y_idx(s, j)] = 1.0
            constraints.append(LinearConstraint(A, 0.0, 1.0))

    # Objective
    c = np.zeros(n_vars)

    tier_big = 100000.0
    diversity_big = 1000.0
    mem_small = 1.0
    food_tiny = 0.01

    # Maximize total tier across slots by adding negative tier cost to selected jobs
    for s in range(S):
        for j in range(J):
            c[y_idx(s, j)] = -tier_big * float(jobs.loc[j, "TierNum"])

    # Memory rarity costs
    if mem_tier_costs is None:
        mem_tier_costs = {"T0": 0.5, "T1": 1.0, "T2": 3.0, "T3": 9.0, "T4": 27.0}
    mem_rarity = np.array([mem_tier_costs.get(t, 1.0) for t in mem["MemoryTier"].tolist()], dtype=float)
    if memory_objective == "weight":
        mem_rarity = mem_rarity * wM

    for s in range(S):
        for k in range(nM):
            c[xM_idx(s, k)] = float(mem_small * mem_rarity[k])

    # Food rarity costs
    if food_tier_costs is None:
        food_tier_costs = {"T1": 1.0, "T2": 3.0, "T3": 9.0, "T4": 27.0}
    food_rarity = np.array([food_tier_costs.get(t, 1.0) for t in food["FoodTier"].tolist()], dtype=float)
    for s in range(S):
        for i in range(nF):
            c[xF_idx(s, i)] = float(food_tiny * food_rarity[i])

    # Optional soft diversity encouragement when uniqueness is not enforced
    # This is a light nudge: penalize selecting the same career repeatedly by adding a small cost per career per slot,
    # implemented indirectly by encouraging spread via negative tier only is usually enough when unique_careers is on.
    # If unique_careers is off, this stays mild and you can keep it simple.

    res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints)

    if res.status != 0:
        return None, f"Small batch solver infeasible or failed, status {res.status}. Try turning off uniqueness, lowering seeds, or increasing memory inventory."

    x = res.x

    # Decode chosen jobs per slot
    chosen = []
    for s in range(S):
        yvals = np.array([x[y_idx(s, j)] for j in range(J)])
        jstar = int(np.argmax(yvals))
        row = jobs.loc[jstar]

        used_mem = {}
        for k in range(nM):
            val = int(round(x[xM_idx(s, k)]))
            if val > 0:
                used_mem[mem_names[k]] = val

        used_food = {}
        for i in range(nF):
            val = int(round(x[xF_idx(s, i)]))
            if val > 0:
                used_food[food_names[i]] = val

        chosen.append({
            "Slot": s + 1,
            "Career": row["Career"],
            "Job": row["Job"],
            "Tier": row["Tier"],
            "TierNum": int(row["TierNum"]),
            "MemUsed": used_mem,
            "FoodUsed": used_food
        })

    # Aggregates
    mem_agg = {}
    food_agg = {}
    tier_counts = {}

    for r in chosen:
        tier_counts[r["TierNum"]] = tier_counts.get(r["TierNum"], 0) + 1
        for k, v in r["MemUsed"].items():
            mem_agg[k] = mem_agg.get(k, 0) + int(v)
        for k, v in r["FoodUsed"].items():
            food_agg[k] = food_agg.get(k, 0) + int(v)

    summary = {
        "SeedsRequested": int(S),
        "TotalHumansPlanned": int(len(chosen)),
        "TierCounts": dict(sorted(tier_counts.items(), key=lambda z: -z[0])),
        "TotalMemoryItemsUsed": int(sum(mem_agg.values())),
        "TotalFoodItemsRequired": int(sum(food_agg.values()))
    }

    return {
        "summary": summary,
        "slots": sorted(chosen, key=lambda r: (-r["TierNum"], r["Slot"])),
        "memory_required": mem_agg,
        "food_required": food_agg
    }, None


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

st.subheader("Settings")
c1, c2, c3, c4 = st.columns(4)
with c1:
    capacity = st.number_input("Rack ingredient capacity, Kg", min_value=1.0, value=float(DEFAULT_RACK_CAPACITY), step=1.0)
with c2:
    seeds_to_use = st.number_input("Seeds to plan for", min_value=1, value=3, step=1)
with c3:
    memory_objective = st.selectbox("Memory economy rule", options=["count", "weight"], index=0)
with c4:
    mode = st.selectbox("Planner mode", options=["Small batch, slot based (best for 1 to 12)", "Campaign, aggregate (best for 20 to 85)"], index=0)

st.subheader("Rarity costs")
st.caption("Lower cost means the solver prefers using it.")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    food_t1 = st.number_input("Food T1 cost", min_value=0.0, value=1.0, step=0.5)
with fc2:
    food_t2 = st.number_input("Food T2 cost", min_value=0.0, value=3.0, step=0.5)
with fc3:
    food_t3 = st.number_input("Food T3 cost", min_value=0.0, value=9.0, step=0.5)
with fc4:
    food_t4 = st.number_input("Food T4 cost", min_value=0.0, value=27.0, step=0.5)

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
with mc1:
    mem_t0 = st.number_input("Memory T0 cost", min_value=0.0, value=0.5, step=0.5)
with mc2:
    mem_t1 = st.number_input("Memory T1 cost", min_value=0.0, value=1.0, step=0.5)
with mc3:
    mem_t2 = st.number_input("Memory T2 cost", min_value=0.0, value=3.0, step=0.5)
with mc4:
    mem_t3 = st.number_input("Memory T3 cost", min_value=0.0, value=9.0, step=0.5)
with mc5:
    mem_t4 = st.number_input("Memory T4 cost", min_value=0.0, value=27.0, step=0.5)

food_costs = {"T1": food_t1, "T2": food_t2, "T3": food_t3, "T4": food_t4}
mem_costs = {"T0": mem_t0, "T1": mem_t1, "T2": mem_t2, "T3": mem_t3, "T4": mem_t4}
food_cost_map, mem_cost_map = costs_from_inputs(food_costs, mem_costs)

st.subheader("Memory inventory")
inv_mem = {}
mem_names = mem_df["Memory"].tolist()
left, right = st.columns(2)
half = int(np.ceil(len(mem_names) / 2))

with left:
    for name in mem_names[:half]:
        inv_mem[name] = st.number_input(name, min_value=0, value=0, step=1, key=f"mem_{name}_l")
with right:
    for name in mem_names[half:]:
        inv_mem[name] = st.number_input(name, min_value=0, value=0, step=1, key=f"mem_{name}_r")

if mode.startswith("Small batch"):
    st.subheader("Small batch options")
    o1, o2, o3 = st.columns(3)
    with o1:
        unique_careers = st.checkbox("No duplicate careers in the batch", value=True)
    with o2:
        unique_jobs = st.checkbox("No duplicate jobs in the batch", value=False)
    with o3:
        st.caption("If no duplicate careers is on, a batch of 3 will not repeat the same career.")

    if st.button("Solve small batch plan"):
        with st.spinner("Solving small batch..."):
            result, err = small_batch_slot_solver(
                careers_df, food_df, mem_df,
                inv_mem=inv_mem,
                seeds_to_use=int(seeds_to_use),
                capacity_kg=float(capacity),
                unique_careers=unique_careers,
                unique_jobs=unique_jobs,
                maximize_career_diversity=True,
                memory_objective=memory_objective,
                food_tier_costs=food_cost_map,
                mem_tier_costs=mem_cost_map
            )

        if err:
            st.warning(err)
        else:
            st.success("Small batch plan solved.")
            st.write(result["summary"])

            slot_df = pd.DataFrame([{
                "Slot": r["Slot"],
                "Launch": (r["Slot"] - 1) // 3 + 1,
                "Pod": (r["Slot"] - 1) % 3 + 1,
                "TierNum": r["TierNum"],
                "Tier": r["Tier"],
                "Career": r["Career"],
                "Job": r["Job"]
            } for r in result["slots"]]).sort_values(["Launch", "Pod"])
            st.dataframe(slot_df, use_container_width=True)

            st.markdown("Per slot recipes")
            for r in sorted(result["slots"], key=lambda z: z["Slot"]):
                st.markdown(f"Launch {(r['Slot'] - 1)//3 + 1}, Pod {(r['Slot'] - 1)%3 + 1}, Tier {r['Tier']}, {r['Job']}, {r['Career']}")
                cA, cB = st.columns(2)
                with cA:
                    st.write("Memories", r["MemUsed"])
                with cB:
                    st.write("Food", r["FoodUsed"])

            st.markdown("Total required Memory items")
            st.dataframe(df_from_dict(result["memory_required"], "Memory", "Count"), use_container_width=True)

            st.markdown("Total required Food items")
            st.dataframe(df_from_dict(result["food_required"], "Food", "Count"), use_container_width=True)

else:
    st.subheader("Campaign planner")
    enforce_div = st.checkbox("Diversity, at least one top tier per Career category", value=False)

    if st.button("Solve campaign plan"):
        with st.spinner("Solving campaign..."):
            result, err = aggregate_campaign_solver(
                careers_df, food_df, mem_df,
                inv_mem=inv_mem,
                seeds_to_use=int(seeds_to_use),
                capacity_kg=float(capacity),
                enforce_diversity=enforce_div,
                memory_objective=memory_objective,
                food_tier_costs=food_cost_map,
                mem_tier_costs=mem_cost_map
            )

        if err:
            st.warning(err)
        else:
            st.success("Campaign plan solved.")
            st.write(result["summary"])

            job_df = pd.DataFrame([{
                "TierNum": r["TierNum"],
                "Tier": r["Tier"],
                "Career": r["Career"],
                "Job": r["Job"],
                "Humans": r["Humans"]
            } for r in result["plan_rows"]]).sort_values(["TierNum", "Humans"], ascending=[False, False])
            st.dataframe(job_df, use_container_width=True)

            st.markdown("Total required Memory items")
            st.dataframe(df_from_dict(result["memory_required"], "Memory", "Count"), use_container_width=True)

            st.markdown("Total required Food items")
            st.dataframe(df_from_dict(result["food_required"], "Food", "Count"), use_container_width=True)

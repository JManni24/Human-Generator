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
        raise ValueError("Food sheet must have an item name column (expected first column as 'Unnamed: 0' or 'Food').")
    if "Memory" not in mem.columns:
        raise ValueError("Memories sheet must have an item name column (expected first column as 'Unnamed: 0' or 'Memory').")

    food["Food"] = food["Food"].apply(normalize_text)
    mem["Memory"] = mem["Memory"].apply(normalize_text)

    careers["TierNum"] = careers["Tier"].apply(tier_num)

    if "Career" not in careers.columns or "Job" not in careers.columns:
        raise ValueError("Careers sheet must include columns named 'Career' and 'Job'.")

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

    if "ItemWeight" not in food.columns or "ItemWeight" not in mem.columns:
        raise ValueError("Food and Memories sheets must include 'Food Weight (Kg)' and 'Memory Weight (Kg)' columns.")

    food["ItemWeight"] = pd.to_numeric(food["ItemWeight"], errors="coerce").fillna(0)
    mem["ItemWeight"] = pd.to_numeric(mem["ItemWeight"], errors="coerce").fillna(0)

    for col in BIO + MENTAL:
        if col in careers.columns:
            careers[col] = pd.to_numeric(careers[col], errors="coerce")
        else:
            careers[col] = np.nan

    return careers, food, mem


def solve_for_job(job_row, food, mem, inv_food, inv_mem, capacity_kg, memory_objective="count"):
    food_names = food["Food"].tolist()
    mem_names = mem["Memory"].tolist()
    nF, nM = len(food_names), len(mem_names)

    lb = np.zeros(nF + nM)
    ub = np.array(
        [inv_food.get(n, 0) for n in food_names] +
        [inv_mem.get(n, 0) for n in mem_names],
        dtype=float
    )

    w_food = food["ItemWeight"].to_numpy()
    w_mem = mem["ItemWeight"].to_numpy()
    w = np.concatenate([w_food, w_mem])

    if memory_objective == "weight":
        c = np.concatenate([np.zeros(nF), w_mem])
    else:
        c = np.concatenate([np.zeros(nF), np.ones(nM)])

    integrality = np.ones(nF + nM, dtype=int)
    constraints = []

    constraints.append(LinearConstraint(w.reshape(1, -1), -np.inf, capacity_kg))

    for tr in BIO:
        req = job_row.get(tr, np.nan)
        if pd.isna(req):
            req = 0.0
        if tr in GLOBAL_MIN:
            req = max(float(req), float(GLOBAL_MIN[tr]))
        coef = np.concatenate([food[tr].to_numpy(), np.zeros(nM)])
        constraints.append(LinearConstraint(coef, req, np.inf))

    for tr in MENTAL:
        req = job_row.get(tr, np.nan)
        if pd.isna(req):
            continue
        coef = np.concatenate([np.zeros(nF), mem[tr].to_numpy()])
        constraints.append(LinearConstraint(coef, float(req), np.inf))

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints
    )

    if res.status != 0:
        return None

    x = res.x
    used_food = {food_names[i]: int(round(x[i])) for i in range(nF) if x[i] > 1e-6}
    used_mem = {mem_names[j]: int(round(x[nF + j])) for j in range(nM) if x[nF + j] > 1e-6}

    totals = {}
    totals["IngredientWeightKg"] = float(np.dot(x, w))
    for tr in BIO:
        totals[tr] = float(np.dot(x[:nF], food[tr].to_numpy()))
    for tr in MENTAL:
        totals[tr] = float(np.dot(x[nF:], mem[tr].to_numpy()))

    return {"used_food": used_food, "used_mem": used_mem, "totals": totals}


def best_job(careers, food, mem, inv_food, inv_mem, capacity_kg, memory_objective="count"):
    careers_sorted = careers.sort_values(["TierNum"], ascending=False)
    for _, row in careers_sorted.iterrows():
        sol = solve_for_job(row, food, mem, inv_food, inv_mem, capacity_kg, memory_objective=memory_objective)
        if sol is not None:
            return {
                "CareerGroup": row["Career"],
                "Tier": row["Tier"],
                "TierNum": row["TierNum"],
                "Job": row["Job"],
                "solution": sol
            }
    return None


def plan_many_greedy(careers, food, mem, inv_food, inv_mem, seeds_to_use, capacity_kg, memory_objective="count"):
    inv_food = dict(inv_food)
    inv_mem = dict(inv_mem)

    plan = []
    for _ in range(int(seeds_to_use)):
        result = best_job(careers, food, mem, inv_food, inv_mem, capacity_kg, memory_objective=memory_objective)
        if result is None:
            break

        sol = result["solution"]
        plan.append(result)

        for k, v in sol["used_food"].items():
            inv_food[k] = max(0, inv_food.get(k, 0) - v)
        for k, v in sol["used_mem"].items():
            inv_mem[k] = max(0, inv_mem.get(k, 0) - v)

    return plan, inv_food, inv_mem


def true_best_possible_global(careers, food, mem, inv_food, inv_mem, seeds_to_use, capacity_kg,
                              memory_objective="count", enforce_diversity=False, diversity_scope="all"):
    careers = careers.copy()

    jobs = careers.reset_index(drop=True)
    J = len(jobs)

    food_names = food["Food"].tolist()
    mem_names = mem["Memory"].tolist()
    nF, nM = len(food_names), len(mem_names)

    invF = np.array([inv_food.get(n, 0) for n in food_names], dtype=float)
    invM = np.array([inv_mem.get(n, 0) for n in mem_names], dtype=float)

    wF = food["ItemWeight"].to_numpy()
    wM = mem["ItemWeight"].to_numpy()

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
        ub[y_idx(j)] = seeds_to_use

    for j in range(J):
        for i in range(nF):
            ub[xF_idx(j, i)] = invF[i]
        for k in range(nM):
            ub[xM_idx(j, k)] = invM[k]

    integrality = np.ones(n_vars, dtype=int)

    constraints = []

    A = np.zeros((1, n_vars))
    for j in range(J):
        A[0, y_idx(j)] = 1
    constraints.append(LinearConstraint(A, 0, float(seeds_to_use)))

    for i in range(nF):
        A = np.zeros((1, n_vars))
        for j in range(J):
            A[0, xF_idx(j, i)] = 1
        constraints.append(LinearConstraint(A, 0, invF[i]))

    for k in range(nM):
        A = np.zeros((1, n_vars))
        for j in range(J):
            A[0, xM_idx(j, k)] = 1
        constraints.append(LinearConstraint(A, 0, invM[k]))

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
        if diversity_scope == "all":
            career_groups = sorted(jobs["Career"].dropna().unique().tolist())
        else:
            career_groups = diversity_scope

        for cg in career_groups:
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

    tier_weight = 10_000.0
    for j in range(J):
        c[y_idx(j)] = -tier_weight * float(jobs.loc[j, "TierNum"])

    if memory_objective == "weight":
        mem_cost = wM
    else:
        mem_cost = np.ones(nM)

    for j in range(J):
        for k in range(nM):
            c[xM_idx(j, k)] = float(mem_cost[k])

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints
    )

    if res.status != 0:
        return None, f"MILP failed or infeasible. Status {res.status}. Try turning off diversity or reducing seeds."

    x = res.x

    y = np.array([int(round(x[y_idx(j)])) for j in range(J)], dtype=int)
    chosen_jobs = np.where(y > 0)[0].tolist()

    output_rows = []
    for j in chosen_jobs:
        row = jobs.loc[j]
        yj = y[j]

        used_food = {}
        for i in range(nF):
            val = int(round(x[xF_idx(j, i)]))
            if val > 0:
                used_food[food_names[i]] = val

        used_mem = {}
        for k in range(nM):
            val = int(round(x[xM_idx(j, k)]))
            if val > 0:
                used_mem[mem_names[k]] = val

        output_rows.append({
            "Career": row["Career"],
            "Job": row["Job"],
            "Tier": row["Tier"],
            "TierNum": row["TierNum"],
            "Humans": yj,
            "FoodUsed": used_food,
            "MemUsed": used_mem
        })

    total_humans = int(y.sum())
    total_mem_items = int(sum(sum(d.values()) for d in [r["MemUsed"] for r in output_rows])) if output_rows else 0
    total_food_items = int(sum(sum(d.values()) for d in [r["FoodUsed"] for r in output_rows])) if output_rows else 0

    summary = {
        "TotalHumansPlanned": total_humans,
        "TotalFoodItemsUsed": total_food_items,
        "TotalMemoryItemsUsed": total_mem_items,
        "SeedsRequested": int(seeds_to_use)
    }

    return {"summary": summary, "plan_rows": output_rows}, None


def render_launches(plan_list):
    rows = []
    for idx, r in enumerate(plan_list, start=1):
        launch = (idx - 1) // 3 + 1
        slot = (idx - 1) % 3 + 1
        rows.append({
            "Launch": launch,
            "Slot": slot,
            "Tier": r["Tier"],
            "Job": r["Job"],
            "Career": r["CareerGroup"]
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Last Caretaker Human Calculator", layout="wide")
st.title("Last Caretaker Human Calculator")

uploaded = st.file_uploader("Upload Human Calculator.xlsx", type=["xlsx"])
if uploaded is None:
    st.info("Upload your Excel file to begin.")
    st.stop()

try:
    careers_df, food_df, mem_df = load_tables_from_excel(uploaded)
except Exception as e:
    st.error(f"Failed to load workbook: {e}")
    st.stop()

st.subheader("Settings")
colA, colB, colC = st.columns(3)
with colA:
    capacity = st.number_input("Lazarus Rack ingredient weight capacity, Kg", min_value=1.0, value=float(DEFAULT_RACK_CAPACITY), step=1.0)
with colB:
    seeds_to_use = st.number_input("How many Human Seeds you want to plan for", min_value=1, value=3, step=1)
with colC:
    memory_objective = st.selectbox("Economy rule for memory items", options=["count", "weight"], index=0)

st.subheader("Inventory")
left, right = st.columns(2)
inv_food = {}
inv_mem = {}

with left:
    st.markdown("Food items on hand")
    for name in food_df["Food"].tolist():
        inv_food[name] = st.number_input(name, min_value=0, value=0, step=1, key=f"food_{name}")

with right:
    st.markdown("Memory items on hand")
    for name in mem_df["Memory"].tolist():
        inv_mem[name] = st.number_input(name, min_value=0, value=0, step=1, key=f"mem_{name}")

st.subheader("Single best human, memory economical")
if st.button("Find highest tier profession for one seed"):
    with st.spinner("Solving..."):
        result = best_job(careers_df, food_df, mem_df, inv_food, inv_mem, capacity, memory_objective=memory_objective)

    if result is None:
        st.warning("No careers are craftable with the current inventory and constraints.")
    else:
        st.success(f"Best craftable job: Tier {result['Tier']}   {result['Job']}   {result['CareerGroup']}")
        sol = result["solution"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Selected Food Items")
            st.dataframe(pd.DataFrame(list(sol["used_food"].items()), columns=["Food", "Count"]) if sol["used_food"] else pd.DataFrame(columns=["Food", "Count"]))
        with c2:
            st.markdown("Selected Memory Items")
            st.dataframe(pd.DataFrame(list(sol["used_mem"].items()), columns=["Memory", "Count"]) if sol["used_mem"] else pd.DataFrame(columns=["Memory", "Count"]))

        st.markdown("Totals")
        st.dataframe(pd.DataFrame(list(sol["totals"].items()), columns=["Stat", "Value"]))

st.subheader("Greedy campaign plan across seeds, practical and fast")
if st.button("Plan across seeds, greedy"):
    with st.spinner("Planning..."):
        plan, inv_food_left, inv_mem_left = plan_many_greedy(
            careers_df, food_df, mem_df,
            inv_food, inv_mem,
            seeds_to_use, capacity,
            memory_objective=memory_objective
        )

    if not plan:
        st.warning("No craftable humans with current inventory.")
    else:
        st.success(f"Planned {len(plan)} humans.")
        st.dataframe(render_launches(plan))

        st.markdown("Remaining inventory after plan")
        remF = pd.DataFrame(sorted(inv_food_left.items()), columns=["Food", "Remaining"])
        remM = pd.DataFrame(sorted(inv_mem_left.items()), columns=["Memory", "Remaining"])
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(remF)
        with c2:
            st.dataframe(remM)

st.subheader("True best possible across seeds, global MILP")
div_col1, div_col2 = st.columns([1, 3])
with div_col1:
    enforce_div = st.checkbox("Enforce diversity", value=False)
with div_col2:
    st.caption("Diversity means at least one top tier human from each Career category. If infeasible, turn it off or lower seeds.")

if st.button("Solve true best possible globally"):
    with st.spinner("Global optimization running..."):
        global_result, err = true_best_possible_global(
            careers_df, food_df, mem_df,
            inv_food, inv_mem,
            int(seeds_to_use), float(capacity),
            memory_objective=memory_objective,
            enforce_diversity=enforce_div,
            diversity_scope="all"
        )

    if err is not None:
        st.warning(err)
    else:
        st.success("Global plan solved.")
        st.write(global_result["summary"])

        plan_rows = global_result["plan_rows"]
        df = pd.DataFrame([{
            "Career": r["Career"],
            "Job": r["Job"],
            "Tier": r["Tier"],
            "Humans": r["Humans"]
        } for r in plan_rows]).sort_values(["Humans", "Tier"], ascending=[False, False])

        st.markdown("Humans by job")
        st.dataframe(df, use_container_width=True)

        st.markdown("Ingredients by job")
        for r in sorted(plan_rows, key=lambda x: (-x["TierNum"], x["Career"], x["Job"])):
            st.markdown(f"Career {r['Career']}   Tier {r['Tier']}   Job {r['Job']}   Humans {r['Humans']}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("Food used", r["FoodUsed"])
            with c2:
                st.write("Memory used", r["MemUsed"])

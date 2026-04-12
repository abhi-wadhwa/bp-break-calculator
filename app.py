#!/usr/bin/env python3
"""
BP Break Calculator — Streamlit App
Monte Carlo break simulator for British Parliamentary debate tournaments.
Supports Calicotab/Tabbycat URL scraping and CSV upload.

Run: streamlit run bp_break_app.py
"""

import json
import random
import re
import ssl
import time
from collections import defaultdict
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BP Break Calculator",
    page_icon="🏆",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────

BP_POINTS = [3, 2, 1, 0]  # 1st=3, 2nd=2, 3rd=1, 4th=0


# ─── Scraping ─────────────────────────────────────────────────────────────────

def scrape_tabbycat(url: str) -> tuple[list[dict], str]:
    """
    Scrape standings from a Tabbycat/Calicotab standings page.
    Returns (teams_list, tournament_name).
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=15, context=ctx) as resp:
        html = resp.read().decode("utf-8")

    # Try to get tournament name from <title>
    title_match = re.search(r"<title>(.*?)</title>", html)
    tournament_name = ""
    if title_match:
        raw_title = title_match.group(1)
        # Tabbycat titles are like "Tournament Name | Standings"
        tournament_name = raw_title.split("|")[0].split("—")[0].strip()
        tournament_name = tournament_name.replace("Tabbycat", "").strip(" |–—")

    # Extract embedded JSON
    match = re.search(r"tablesData:\s*(\[.+\])\s*\}", html, re.DOTALL)
    if not match:
        raise ValueError(
            "Could not find standings data in page HTML. "
            "The page may require JavaScript rendering or has a different format."
        )

    tables_data = json.loads(match.group(1))
    table = tables_data[0]

    # Detect how many rounds are shown
    round_cols = [h for h in table["head"] if h.get("key", "").startswith("r")]
    num_rounds_shown = len(round_cols)

    teams = []
    for row in table["data"]:
        team_name = row[0].get("text", "Unknown")
        points = int(row[1].get("sort", row[1].get("text", 0)))

        rounds_played = 0
        round_results = []
        for i in range(2, len(row)):
            text = row[i].get("text", "—")
            round_results.append(text)
            if text != "—":
                rounds_played += 1

        popover = row[0].get("popover", {})
        content = popover.get("content", [])
        speakers = content[0].get("text", "") if content else ""

        teams.append({
            "team": team_name,
            "points": points,
            "speaks": 0.0,
            "speakers": speakers,
            "rounds_played": rounds_played,
        })

    # Filter out swing/dropped teams
    min_rounds = max(1, num_rounds_shown // 2)
    active = [
        t for t in teams
        if t["rounds_played"] >= min_rounds and "Swing" not in t["team"]
    ]
    active.sort(key=lambda t: (-t["points"],))

    return active, tournament_name, num_rounds_shown


def parse_csv(uploaded_file) -> list[dict]:
    """Parse uploaded CSV into team list."""
    df = pd.read_csv(uploaded_file)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "team" not in df.columns or "points" not in df.columns:
        raise ValueError("CSV must have 'team' and 'points' columns.")

    teams = []
    for _, row in df.iterrows():
        teams.append({
            "team": str(row["team"]).strip(),
            "points": int(row["points"]),
            "speaks": float(row.get("speaks", 0)),
        })

    teams.sort(key=lambda t: (-t["points"], -t["speaks"]))
    return teams


# ─── Simulation Engine ───────────────────────────────────────────────────────

def simulate_once(points, num_teams, rounds, rng):
    """Simulate remaining rounds with power-pairing."""
    pts = list(points)
    placements = BP_POINTS[:]

    for _ in range(rounds):
        order = [(i, pts[i], rng.random()) for i in range(num_teams)]
        order.sort(key=lambda x: (-x[1], x[2]))

        for room_start in range(0, num_teams, 4):
            room = order[room_start:room_start + 4]
            rng.shuffle(placements)
            for j, (idx, _, _) in enumerate(room):
                if j < len(placements):
                    pts[idx] += placements[j]

    return pts


def run_simulations(teams, num_sims, rounds_remaining, break_size,
                    novice_set, novice_break_size, seed, progress_bar=None):
    """Run Monte Carlo simulations with optional novice tracking."""
    rng = random.Random(seed)
    num_teams = len(teams)
    base_points = [t["points"] for t in teams]
    team_names = [t["team"] for t in teams]
    is_novice = [t["team"] in novice_set for t in teams]

    break_line_dist = defaultdict(int)
    team_break_counts = defaultdict(int)
    team_points_all = defaultdict(list)

    novice_break_line_dist = defaultdict(int)
    novice_break_counts = defaultdict(int)

    batch_size = max(1, num_sims // 100)

    for sim in range(num_sims):
        final_pts = simulate_once(base_points, num_teams, rounds_remaining, rng)

        standings = [(i, final_pts[i], rng.random()) for i in range(num_teams)]
        standings.sort(key=lambda x: (-x[1], x[2]))

        # Open break
        if len(standings) >= break_size:
            break_line_dist[standings[break_size - 1][1]] += 1

        open_breakers = set()
        for rank, (idx, pts, _) in enumerate(standings):
            name = team_names[idx]
            team_points_all[name].append(pts)
            if rank < break_size:
                team_break_counts[name] += 1
                open_breakers.add(idx)

        # Novice break
        if novice_set:
            novice_eligible = []
            for rank, (idx, pts, _) in enumerate(standings):
                if is_novice[idx] and idx not in open_breakers:
                    novice_eligible.append((idx, pts))
                    if len(novice_eligible) == novice_break_size:
                        break

            if novice_eligible:
                novice_break_line_dist[novice_eligible[-1][1]] += 1

            novice_breaking = set(idx for idx, _ in novice_eligible)
            for idx in range(num_teams):
                if is_novice[idx] and (idx in open_breakers or idx in novice_breaking):
                    novice_break_counts[team_names[idx]] += 1

        # Progress
        if progress_bar and (sim + 1) % batch_size == 0:
            progress_bar.progress((sim + 1) / num_sims)

    if progress_bar:
        progress_bar.progress(1.0)

    return {
        "break_line_dist": dict(break_line_dist),
        "team_break_counts": dict(team_break_counts),
        "team_points_all": dict(team_points_all),
        "novice_break_line_dist": dict(novice_break_line_dist),
        "novice_break_counts": dict(novice_break_counts),
        "num_sims": num_sims,
    }


# ─── UI Helpers ───────────────────────────────────────────────────────────────

def color_break_pct(val):
    """Return background color based on break percentage."""
    if val >= 95:
        return "background-color: #2d6a4f; color: white"
    elif val >= 75:
        return "background-color: #40916c; color: white"
    elif val >= 50:
        return "background-color: #52b788; color: white"
    elif val >= 25:
        return "background-color: #f4a261; color: black"
    elif val > 0:
        return "background-color: #e76f51; color: white"
    else:
        return "background-color: #343a40; color: #6c757d"


def make_break_table(teams, results, break_size):
    """Build a DataFrame for the break probability table."""
    num_sims = results["num_sims"]
    rows = []
    for t in teams:
        name = t["team"]
        pts_list = results["team_points_all"].get(name, [])
        prob = results["team_break_counts"].get(name, 0) / num_sims * 100
        avg = np.mean(pts_list) if pts_list else 0
        mn = min(pts_list) if pts_list else 0
        mx = max(pts_list) if pts_list else 0
        rows.append({
            "Team": name,
            "Current Pts": t["points"],
            "Avg Final": round(avg, 1),
            "Min": mn,
            "Max": mx,
            "Break %": round(prob, 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Break %", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


def make_novice_table(teams, results, novice_set):
    """Build a DataFrame for novice break probabilities."""
    num_sims = results["num_sims"]
    rows = []
    for t in teams:
        name = t["team"]
        if name not in novice_set:
            continue
        pts_list = results["team_points_all"].get(name, [])
        open_prob = results["team_break_counts"].get(name, 0) / num_sims * 100
        any_prob = results["novice_break_counts"].get(name, 0) / num_sims * 100
        nov_prob = any_prob - open_prob
        avg = np.mean(pts_list) if pts_list else 0
        rows.append({
            "Team": name,
            "Current Pts": t["points"],
            "Avg Final": round(avg, 1),
            "Open Break %": round(open_prob, 1),
            "Novice Break %": round(nov_prob, 1),
            "Any Break %": round(any_prob, 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Any Break %", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.title("🏆 BP Break Calculator")
    st.caption("Monte Carlo break simulator for British Parliamentary debate tournaments")

    # ── Sidebar: Data Input ───────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Data Source")

        source = st.radio(
            "How to load standings:",
            ["Calicotab / Tabbycat URL", "Upload CSV"],
            index=0,
        )

        teams = None
        tournament_name = "Tournament"
        rounds_completed = 5

        if source == "Calicotab / Tabbycat URL":
            url = st.text_input(
                "Standings URL",
                placeholder="https://tournament.calicotab.com/_/tab/current-standings/",
            )
            if url:
                with st.spinner("Scraping standings..."):
                    try:
                        teams, tournament_name, rounds_completed = scrape_tabbycat(url)
                        st.success(f"Loaded {len(teams)} teams ({rounds_completed} rounds)")
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")

        else:
            st.markdown("""
            CSV format: `team,points` (optional: `speaks`)
            """)
            uploaded = st.file_uploader("Upload standings CSV", type=["csv"])
            if uploaded:
                try:
                    teams = parse_csv(uploaded)
                    st.success(f"Loaded {len(teams)} teams")
                except Exception as e:
                    st.error(f"CSV parse error: {e}")

            rounds_completed = st.number_input(
                "Rounds completed", min_value=1, max_value=20, value=5
            )

        st.divider()

        # ── Sidebar: Simulation Config ────────────────────────────────────
        st.header("⚙️ Configuration")

        total_rounds = st.number_input(
            "Total rounds in tournament",
            min_value=rounds_completed + 1,
            max_value=20,
            value=max(rounds_completed + 2, 7),
        )
        rounds_remaining = total_rounds - rounds_completed

        break_size = st.number_input(
            "Open break size (teams)",
            min_value=4, max_value=64, value=16, step=4,
        )

        num_sims = st.select_slider(
            "Simulations",
            options=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
            value=500_000,
            format_func=lambda x: f"{x:,}",
        )

        seed_option = st.radio("Random seed", ["Fixed (42)", "Random"], index=0)
        seed = 42 if seed_option == "Fixed (42)" else None

        st.divider()

        # ── Sidebar: Novice Teams ─────────────────────────────────────────
        st.header("🌱 Novice Break")

        enable_novice = st.checkbox("Enable novice break analysis", value=False)

        novice_set = set()
        novice_break_size = 4

        if enable_novice and teams:
            novice_break_size = st.number_input(
                "Novice break size",
                min_value=1, max_value=16, value=4,
            )

            team_names = [t["team"] for t in teams]
            novice_selected = st.multiselect(
                "Select novice teams",
                options=team_names,
                default=[],
            )
            novice_set = set(novice_selected)

            if novice_set:
                st.info(f"{len(novice_set)} novice teams selected")

    # ── Main Area ─────────────────────────────────────────────────────────
    if teams is None:
        st.info("👈 Load standings from the sidebar to get started.")
        st.markdown("""
        ### How it works
        1. **Load data** from a Calicotab/Tabbycat URL or CSV
        2. **Configure** tournament parameters (rounds, break size)
        3. **Optionally tag** novice teams for novice break analysis
        4. **Run simulations** to see break probabilities

        The simulator uses **power-paired Monte Carlo simulation**:
        - Teams are sorted by points and grouped into rooms of 4
        - Each room's placements (1st–4th) are assigned randomly
        - This repeats for each remaining round
        - Results from hundreds of thousands of runs give break probabilities
        """)
        return

    # Show current standings
    st.subheader(f"📋 Current Standings — {tournament_name}")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Teams", len(teams))
    col_info2.metric("Rounds Done", rounds_completed)
    col_info3.metric("Rounds Left", rounds_remaining)
    col_info4.metric("Break Size", break_size)

    standings_df = pd.DataFrame([
        {"Team": t["team"], "Points": t["points"]}
        for t in teams
    ])
    standings_df.index = standings_df.index + 1
    standings_df.index.name = "Rank"

    with st.expander("View current standings", expanded=False):
        st.dataframe(standings_df, use_container_width=True, height=400)

    # ── Run Simulation ────────────────────────────────────────────────────
    st.divider()

    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        st.subheader("Running simulations...")

        progress_bar = st.progress(0)
        t0 = time.time()

        results = run_simulations(
            teams=teams,
            num_sims=num_sims,
            rounds_remaining=rounds_remaining,
            break_size=break_size,
            novice_set=novice_set,
            novice_break_size=novice_break_size,
            seed=seed,
            progress_bar=progress_bar,
        )

        elapsed = time.time() - t0
        st.success(f"Completed {num_sims:,} simulations in {elapsed:.1f}s "
                   f"({num_sims / elapsed:,.0f}/s)")

        # Store results in session state so they persist
        st.session_state["results"] = results
        st.session_state["teams"] = teams
        st.session_state["break_size"] = break_size
        st.session_state["novice_set"] = novice_set
        st.session_state["novice_break_size"] = novice_break_size
        st.session_state["tournament_name"] = tournament_name

    # ── Display Results ───────────────────────────────────────────────────
    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    teams = st.session_state["teams"]
    break_size = st.session_state["break_size"]
    novice_set = st.session_state["novice_set"]
    tournament_name = st.session_state["tournament_name"]
    num_sims = results["num_sims"]

    st.divider()

    # ── Break Line Analysis ───────────────────────────────────────────────
    st.subheader("📊 Open Break Line Analysis")

    break_dist = results["break_line_dist"]
    all_lines = sorted(break_dist.keys())
    best = min(all_lines)
    worst = max(all_lines)
    mode = max(break_dist, key=break_dist.get)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Case (easiest)", f"{best} pts")
    col2.metric("Most Likely", f"{mode} pts")
    col3.metric("Worst Case (hardest)", f"{worst} pts")

    # Break line distribution chart
    dist_df = pd.DataFrame([
        {"Points": pts, "Probability (%)": break_dist[pts] / num_sims * 100}
        for pts in all_lines
    ])

    fig = px.bar(
        dist_df, x="Points", y="Probability (%)",
        title=f"Break Line Distribution ({break_size}th place points after all rounds)",
        text="Probability (%)",
        color="Probability (%)",
        color_continuous_scale="Greens",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Points needed
    st.markdown("**What points total do you need to break?**")
    pts_needed = []
    for target in all_lines:
        pct = sum(break_dist[p] for p in break_dist if p <= target) / num_sims * 100
        pts_needed.append({"Points": target, "Breaks in % of scenarios": f"{pct:.1f}%"})
    st.dataframe(pd.DataFrame(pts_needed), use_container_width=True, hide_index=True)

    st.divider()

    # ── Team Break Probabilities ──────────────────────────────────────────
    st.subheader("🏅 Team Break Probabilities")

    break_df = make_break_table(teams, results, break_size)

    # Style the dataframe
    def style_row(row):
        pct = row["Break %"]
        styles = [""] * len(row)
        # Color the Break % column
        break_idx = list(row.index).index("Break %")
        styles[break_idx] = color_break_pct(pct)
        return styles

    styled_df = break_df.style.apply(style_row, axis=1).format({
        "Break %": "{:.1f}%",
        "Avg Final": "{:.1f}",
    })

    st.dataframe(styled_df, use_container_width=True, height=600)

    # Break probability by current points chart
    prob_by_pts = break_df.groupby("Current Pts")["Break %"].mean().reset_index()
    prob_by_pts = prob_by_pts.sort_values("Current Pts")

    fig2 = px.bar(
        prob_by_pts, x="Current Pts", y="Break %",
        title="Average Break Probability by Current Points",
        text="Break %",
        color="Break %",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Bubble Watch ──────────────────────────────────────────────────────
    bubble_teams = break_df[
        (break_df["Break %"] > 5) & (break_df["Break %"] < 95)
    ].copy()

    if not bubble_teams.empty:
        st.subheader("🔥 Bubble Watch")

        def bubble_status(pct):
            if pct >= 75:
                return "🟢 LIKELY IN"
            elif pct >= 50:
                return "🟡 LEANING IN"
            elif pct >= 25:
                return "🟠 ON BUBBLE"
            else:
                return "🔴 NEEDS HELP"

        bubble_teams["Status"] = bubble_teams["Break %"].apply(bubble_status)
        st.dataframe(
            bubble_teams[["Team", "Current Pts", "Break %", "Status"]],
            use_container_width=True,
            hide_index=True,
        )

    # ── Novice Break ──────────────────────────────────────────────────────
    if novice_set and results["novice_break_line_dist"]:
        st.divider()
        st.subheader("🌱 Novice Break Analysis")

        novice_dist = results["novice_break_line_dist"]
        novice_lines = sorted(novice_dist.keys())
        novice_best = min(novice_lines)
        novice_worst = max(novice_lines)
        novice_mode = max(novice_dist, key=novice_dist.get)

        st.markdown(
            f"**Novice break** = Top **{st.session_state['novice_break_size']}** "
            f"novice teams **not already breaking open**."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Case", f"{novice_best} pts")
        col2.metric("Most Likely", f"{novice_mode} pts")
        col3.metric("Worst Case", f"{novice_worst} pts")

        # Novice break line chart
        novice_dist_df = pd.DataFrame([
            {"Points": pts, "Probability (%)": novice_dist[pts] / num_sims * 100}
            for pts in novice_lines
        ])

        fig3 = px.bar(
            novice_dist_df, x="Points", y="Probability (%)",
            title="Novice Break Line Distribution",
            text="Probability (%)",
            color="Probability (%)",
            color_continuous_scale="Purples",
        )
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        # Novice team table
        st.markdown("**Novice Team Probabilities**")
        novice_df = make_novice_table(teams, results, novice_set)

        st.dataframe(
            novice_df.style.format({
                "Open Break %": "{:.1f}%",
                "Novice Break %": "{:.1f}%",
                "Any Break %": "{:.1f}%",
                "Avg Final": "{:.1f}",
            }),
            use_container_width=True,
        )

        # Stacked bar chart for novice teams
        novice_chart_data = []
        for _, row in novice_df.iterrows():
            novice_chart_data.append({
                "Team": row["Team"],
                "Open Break": row["Open Break %"],
                "Novice Break": row["Novice Break %"],
                "No Break": 100 - row["Any Break %"],
            })

        ncd = pd.DataFrame(novice_chart_data)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            name="Open Break", x=ncd["Team"], y=ncd["Open Break"],
            marker_color="#2d6a4f",
        ))
        fig4.add_trace(go.Bar(
            name="Novice Break", x=ncd["Team"], y=ncd["Novice Break"],
            marker_color="#95d5b2",
        ))
        fig4.add_trace(go.Bar(
            name="No Break", x=ncd["Team"], y=ncd["No Break"],
            marker_color="#dee2e6",
        ))
        fig4.update_layout(
            barmode="stack",
            title="Novice Teams — Break Pathway Breakdown",
            yaxis_title="Probability (%)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    main()

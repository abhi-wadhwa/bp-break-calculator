#!/usr/bin/env python3
"""
BP Break Calculator — NAUDC 2026
Scrapes Tabbycat/Calicotab standings and runs Monte Carlo simulations
to determine break line probabilities.

Usage:
    python break_calc.py <url>
    python break_calc.py --csv standings.csv
    python break_calc.py <url> --seed random --sims 500000
"""

import argparse
import csv
import json
import random
import re
import ssl
import sys
import time
from collections import defaultdict
from io import StringIO
from urllib.request import urlopen, Request

# ─── Constants ────────────────────────────────────────────────────────────────
BP_POINTS = [3, 2, 1, 0]  # 1st, 2nd, 3rd, 4th
BREAK_SIZE = 16
NOVICE_BREAK_SIZE = 4
TOTAL_ROUNDS = 7
ROUNDS_COMPLETED = 5
ROUNDS_REMAINING = TOTAL_ROUNDS - ROUNDS_COMPLETED

NOVICE_TEAMS = {
    "Black Square", "Blown Leaves", "Cheese", "Chestnut", "Deep Bow",
    "Desktop", "Grapes", "Heart Hands", "Heated Rivals", "Koala",
    "Tools", "Tornado", "Wilted",
}


# ─── Step 1: Scraping ────────────────────────────────────────────────────────

def scrape_tabbycat(url: str) -> list[dict]:
    """
    Scrape standings from a Tabbycat/Calicotab standings page.
    The data is embedded as JSON in window.vueData.tablesData.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=15, context=ctx) as resp:
        html = resp.read().decode("utf-8")

    # Extract embedded JSON from window.vueData = { tablesData: [...] }
    # Use greedy match — the JSON array ends with }] at the end of vueData
    match = re.search(r"tablesData:\s*(\[.+\])\s*\}", html, re.DOTALL)
    if not match:
        print("ERROR: Could not find tablesData in page HTML.", file=sys.stderr)
        print("The page may require JavaScript rendering or has a different format.",
              file=sys.stderr)
        sys.exit(1)

    tables_data = json.loads(match.group(1))
    table = tables_data[0]

    # Identify column keys
    head = table["head"]
    col_keys = [h.get("key", "") for h in head]

    teams = []
    for row in table["data"]:
        team_name = row[0].get("text", "Unknown")
        points = int(row[1].get("sort", row[1].get("text", 0)))

        # Count rounds actually played (non-dash results)
        rounds_played = 0
        round_results = []
        for i in range(2, len(row)):
            text = row[i].get("text", "—")
            round_results.append(text)
            if text != "—":
                rounds_played += 1

        # Extract speakers from popover
        popover = row[0].get("popover", {})
        content = popover.get("content", [])
        speakers = content[0].get("text", "") if content else ""

        teams.append({
            "team": team_name,
            "points": points,
            "speaks": 0.0,  # Not available on this page
            "speakers": speakers,
            "rounds_played": rounds_played,
            "round_results": round_results,
        })

    # Filter out swing teams and teams that dropped (played < half of rounds)
    min_rounds = max(1, ROUNDS_COMPLETED // 2)
    active_teams = [
        t for t in teams
        if t["rounds_played"] >= min_rounds and "Swing" not in t["team"]
    ]

    # Sort by points desc
    active_teams.sort(key=lambda t: (-t["points"],))

    return active_teams


def load_csv(path: str) -> list[dict]:
    """Load standings from CSV with columns: team, points, speaks (optional)."""
    teams = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams.append({
                "team": row["team"],
                "points": int(row["points"]),
                "speaks": float(row.get("speaks", 0)),
            })
    teams.sort(key=lambda t: (-t["points"], -t["speaks"]))
    return teams


# ─── Step 3: Power Matching ──────────────────────────────────────────────────

def power_match(team_points: list[tuple[int, int]]) -> list[list[int]]:
    """
    Power-match teams into rooms of 4.

    Args:
        team_points: list of (team_index, points) sorted by points desc
                     with random shuffling within tied groups already applied.

    Returns:
        List of rooms, each room is a list of 4 team indices.
    """
    rooms = []
    for i in range(0, len(team_points), 4):
        group = team_points[i:i + 4]
        indices = [t[0] for t in group]
        # Pad with -1 (swing) if fewer than 4
        while len(indices) < 4:
            indices.append(-1)
        rooms.append(indices)
    return rooms


# ─── Step 4: Simulation Engine ───────────────────────────────────────────────

def simulate_once(
    points: list[int],
    num_teams: int,
    rounds: int,
    rng: random.Random,
) -> list[int]:
    """
    Simulate `rounds` remaining rounds for all teams.

    Args:
        points: current points for each team (will be copied).
        num_teams: number of teams.
        rounds: number of rounds to simulate.
        rng: random.Random instance for reproducibility.

    Returns:
        Final points list.
    """
    pts = list(points)  # working copy

    for _ in range(rounds):
        # Build sortable list: (team_index, points, random_tiebreak)
        order = [(i, pts[i], rng.random()) for i in range(num_teams)]
        # Sort by points desc, then random tiebreak (simulates speaks tiebreaker)
        order.sort(key=lambda x: (-x[1], x[2]))

        # Power match into rooms of 4
        placements = BP_POINTS[:]  # [3, 2, 1, 0]

        for room_start in range(0, num_teams, 4):
            room = order[room_start:room_start + 4]

            if len(room) < 4:
                # Incomplete room: pad with phantom teams, still assign points
                rng.shuffle(placements)
                for j, (idx, _, _) in enumerate(room):
                    pts[idx] += placements[j]
            else:
                rng.shuffle(placements)
                for j in range(4):
                    pts[room[j][0]] += placements[j]

    return pts


def run_simulations(
    teams: list[dict],
    num_sims: int,
    seed: int | None = 42,
) -> dict:
    """
    Run Monte Carlo break simulations.

    Returns dict with:
        - break_line_dist: {points: count}
        - team_break_counts: {team_name: count}
        - team_points_sums: {team_name: total_points_across_sims}  (for avg)
        - team_points_min: {team_name: min}
        - team_points_max: {team_name: max}
    """
    rng = random.Random(seed)
    num_teams = len(teams)
    base_points = [t["points"] for t in teams]
    team_names = [t["team"] for t in teams]
    is_novice = [t["team"] in NOVICE_TEAMS for t in teams]

    break_line_dist = defaultdict(int)
    team_break_counts = defaultdict(int)
    team_points_sums = defaultdict(float)
    team_points_min = {name: 999 for name in team_names}
    team_points_max = {name: 0 for name in team_names}

    # Novice break tracking
    novice_break_line_dist = defaultdict(int)
    novice_break_counts = defaultdict(int)  # novice teams that break novice

    t0 = time.time()

    for sim in range(num_sims):
        final_pts = simulate_once(base_points, num_teams, ROUNDS_REMAINING, rng)

        # Sort final standings: points desc, random tiebreak for speaks
        standings = [
            (i, final_pts[i], rng.random()) for i in range(num_teams)
        ]
        standings.sort(key=lambda x: (-x[1], x[2]))

        # Record 16th place points (open break line)
        if len(standings) >= BREAK_SIZE:
            break_line = standings[BREAK_SIZE - 1][1]
            break_line_dist[break_line] += 1

        # Determine open breakers (top 16)
        open_breakers = set()
        for rank, (idx, pts, _) in enumerate(standings):
            if rank < BREAK_SIZE:
                open_breakers.add(idx)

        # Novice break: top NOVICE_BREAK_SIZE novice teams NOT in open break
        novice_eligible = []
        for rank, (idx, pts, _) in enumerate(standings):
            if is_novice[idx] and idx not in open_breakers:
                novice_eligible.append((idx, pts))
                if len(novice_eligible) == NOVICE_BREAK_SIZE:
                    break

        if len(novice_eligible) == NOVICE_BREAK_SIZE:
            novice_line = novice_eligible[-1][1]  # 4th novice team's points
            novice_break_line_dist[novice_line] += 1
        elif novice_eligible:
            # Fewer than 4 novice teams outside open break (some broke open)
            novice_line = novice_eligible[-1][1]
            novice_break_line_dist[novice_line] += 1

        # Track which novice teams break novice
        novice_breaking = set(idx for idx, _ in novice_eligible)
        for idx in range(num_teams):
            if is_novice[idx]:
                if idx in open_breakers or idx in novice_breaking:
                    novice_break_counts[team_names[idx]] += 1

        # Record per-team results
        for rank, (idx, pts, _) in enumerate(standings):
            name = team_names[idx]
            team_points_sums[name] += pts
            if pts < team_points_min[name]:
                team_points_min[name] = pts
            if pts > team_points_max[name]:
                team_points_max[name] = pts
            if rank < BREAK_SIZE:
                team_break_counts[name] += 1

        # Progress update
        if (sim + 1) % 100_000 == 0:
            elapsed = time.time() - t0
            rate = (sim + 1) / elapsed
            print(f"  ... {sim + 1:,} / {num_sims:,} simulations "
                  f"({elapsed:.1f}s, {rate:,.0f}/s)", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"  Completed {num_sims:,} simulations in {elapsed:.1f}s "
          f"({num_sims / elapsed:,.0f}/s)\n", file=sys.stderr)

    return {
        "break_line_dist": dict(break_line_dist),
        "team_break_counts": dict(team_break_counts),
        "team_points_sums": dict(team_points_sums),
        "team_points_min": dict(team_points_min),
        "team_points_max": dict(team_points_max),
        "novice_break_line_dist": dict(novice_break_line_dist),
        "novice_break_counts": dict(novice_break_counts),
        "num_sims": num_sims,
    }


# ─── Step 5 & 6: Output ─────────────────────────────────────────────────────

def print_results(teams: list[dict], results: dict):
    num_sims = results["num_sims"]
    break_dist = results["break_line_dist"]
    break_counts = results["team_break_counts"]
    points_sums = results["team_points_sums"]
    points_min = results["team_points_min"]
    points_max = results["team_points_max"]

    num_teams = len(teams)

    # Header
    print("=" * 72)
    print("  BP BREAK CALCULATOR — NAUDC 2026")
    print("=" * 72)
    print(f"  Teams: {num_teams} | Rounds completed: {ROUNDS_COMPLETED} "
          f"| Rounds remaining: {ROUNDS_REMAINING}")
    print(f"  Break size: Top {BREAK_SIZE} | Simulations: {num_sims:,}")
    print()

    # Break line distribution
    all_break_lines = sorted(break_dist.keys())
    best_case = min(all_break_lines)
    worst_case = max(all_break_lines)
    mode_line = max(break_dist, key=break_dist.get)

    print("─" * 72)
    print("  BREAK LINE (16th place after 7 rounds)")
    print("─" * 72)
    print(f"  Best case  (easiest break):   {best_case} points")
    print(f"  Most likely:                  {mode_line} points")
    print(f"  Worst case (hardest break):   {worst_case} points")
    print()

    # Distribution bar chart
    print("─" * 72)
    print("  BREAK LINE DISTRIBUTION")
    print("─" * 72)
    max_pct = max(break_dist[p] / num_sims * 100 for p in all_break_lines)
    bar_scale = 45 / max_pct if max_pct > 0 else 1

    for pts in all_break_lines:
        pct = break_dist[pts] / num_sims * 100
        bar_len = int(pct * bar_scale)
        bar = "█" * bar_len
        print(f"  {pts:>2} pts | {pct:>6.2f}% | {bar}")
    print()

    # Per-team break probabilities
    print("─" * 72)
    print("  TEAM BREAK PROBABILITIES")
    print("─" * 72)
    print(f"  {'Rank':>4}  {'Team':<44} {'Pts':>3}  {'Avg':>5}"
          f"  {'Min':>3}  {'Max':>3}  {'Break%':>7}")
    print("  " + "-" * 68)

    # Sort teams by break probability desc, then current points desc
    team_results = []
    for t in teams:
        name = t["team"]
        prob = break_counts.get(name, 0) / num_sims * 100
        avg = points_sums.get(name, 0) / num_sims
        mn = points_min.get(name, 0)
        mx = points_max.get(name, 0)
        team_results.append((name, t["points"], prob, avg, mn, mx))

    team_results.sort(key=lambda x: (-x[2], -x[1]))

    for rank, (name, curr, prob, avg, mn, mx) in enumerate(team_results, 1):
        # Status indicator
        if prob >= 99:
            marker = "✅"
        elif prob >= 90:
            marker = "🟢"
        elif prob >= 75:
            marker = "🔵"
        elif prob >= 50:
            marker = "🟡"
        elif prob >= 25:
            marker = "🟠"
        elif prob > 0:
            marker = "🔴"
        else:
            marker = "⬛"

        # Truncate long names
        display_name = name[:42] if len(name) > 42 else name

        print(f"  {rank:>4}  {marker} {display_name:<42} {curr:>3}  {avg:>5.1f}"
              f"  {mn:>3}  {mx:>3}  {prob:>6.2f}%")

        # Visual separator between break and non-break zone
        if rank == BREAK_SIZE:
            print("  " + "·" * 68 + "  ← break line")

    # Points needed analysis
    print()
    print("─" * 72)
    print("  WHAT POINTS TOTAL DO YOU NEED?")
    print("─" * 72)
    # For each possible final points total, what % of sims had that as enough
    for target in sorted(break_dist.keys()):
        # % of sims where break line was ≤ target (i.e., target pts was enough)
        pct_enough = sum(
            break_dist[p] for p in break_dist if p <= target
        ) / num_sims * 100
        print(f"  {target:>2} pts: breaks in {pct_enough:>6.2f}% of scenarios")
    print()

    # Bubble teams analysis
    print("─" * 72)
    print("  BUBBLE WATCH")
    print("─" * 72)
    bubble = [
        (name, curr, prob) for name, curr, prob, _, _, _ in team_results
        if 5 < prob < 95
    ]
    if bubble:
        for name, curr, prob in bubble:
            status = "LIKELY IN" if prob >= 50 else "ON BUBBLE" if prob >= 25 else "NEEDS HELP"
            print(f"  {name:<44} {curr:>3} pts  {prob:>5.1f}%  {status}")
    else:
        print("  No teams in the bubble zone (5%-95%).")
    print()

    # ─── Novice Break Analysis ────────────────────────────────────────────
    novice_dist = results.get("novice_break_line_dist", {})
    novice_bcounts = results.get("novice_break_counts", {})

    if not novice_dist:
        return

    novice_teams_list = [t for t in teams if t["team"] in NOVICE_TEAMS]

    print("=" * 72)
    print("  NOVICE BREAK ANALYSIS")
    print("=" * 72)
    print(f"  Novice teams: {len(novice_teams_list)} | "
          f"Novice break: Top {NOVICE_BREAK_SIZE} not breaking open")
    print(f"  (Novice teams CAN break open — novice break is for the rest)")
    print()

    # Novice break line distribution
    all_novice_lines = sorted(novice_dist.keys())
    novice_best = min(all_novice_lines)
    novice_worst = max(all_novice_lines)
    novice_mode = max(novice_dist, key=novice_dist.get)

    print("─" * 72)
    print(f"  NOVICE BREAK LINE (4th novice team not breaking open)")
    print("─" * 72)
    print(f"  Best case  (easiest break):   {novice_best} points")
    print(f"  Most likely:                  {novice_mode} points")
    print(f"  Worst case (hardest break):   {novice_worst} points")
    print()

    # Distribution
    print("─" * 72)
    print("  NOVICE BREAK LINE DISTRIBUTION")
    print("─" * 72)
    max_pct = max(novice_dist[p] / num_sims * 100 for p in all_novice_lines)
    bar_scale = 45 / max_pct if max_pct > 0 else 1

    for pts in all_novice_lines:
        pct = novice_dist[pts] / num_sims * 100
        bar_len = int(pct * bar_scale)
        bar = "█" * bar_len
        print(f"  {pts:>2} pts | {pct:>6.2f}% | {bar}")
    print()

    # Per novice team probabilities
    print("─" * 72)
    print("  NOVICE TEAM PROBABILITIES")
    print("─" * 72)
    print(f"  {'Team':<44} {'Pts':>3}  {'Avg':>5}  "
          f"{'Open%':>6}  {'Nov%':>6}  {'Any%':>6}")
    print("  " + "-" * 68)

    novice_results = []
    for t in novice_teams_list:
        name = t["team"]
        curr = t["points"]
        open_prob = break_counts.get(name, 0) / num_sims * 100
        any_prob = novice_bcounts.get(name, 0) / num_sims * 100
        novice_only_prob = any_prob - open_prob
        avg = points_sums.get(name, 0) / num_sims
        novice_results.append((name, curr, avg, open_prob, novice_only_prob, any_prob))

    novice_results.sort(key=lambda x: (-x[5], -x[1]))

    for name, curr, avg, open_pct, nov_pct, any_pct in novice_results:
        if any_pct >= 99:
            marker = "✅"
        elif any_pct >= 90:
            marker = "🟢"
        elif any_pct >= 75:
            marker = "🔵"
        elif any_pct >= 50:
            marker = "🟡"
        elif any_pct >= 25:
            marker = "🟠"
        elif any_pct > 0:
            marker = "🔴"
        else:
            marker = "⬛"

        display_name = name[:42] if len(name) > 42 else name
        print(f"  {marker} {display_name:<43} {curr:>3}  {avg:>5.1f}  "
              f"{open_pct:>5.1f}%  {nov_pct:>5.1f}%  {any_pct:>5.1f}%")
    print()
    print("  Open% = breaks in top 16 overall")
    print("  Nov%  = breaks novice (not open)")
    print("  Any%  = breaks via either path")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    global BREAK_SIZE, ROUNDS_REMAINING

    parser = argparse.ArgumentParser(
        description="BP Break Calculator for Tabbycat tournaments"
    )
    parser.add_argument("url", nargs="?", help="Tabbycat standings URL")
    parser.add_argument("--csv", help="CSV fallback file (columns: team, points, speaks)")
    parser.add_argument("--sims", type=int, default=500_000,
                        help="Number of simulations (default: 500,000)")
    parser.add_argument("--seed", default="42",
                        help="Random seed (use 'random' for non-deterministic)")
    parser.add_argument("--break-size", type=int, default=BREAK_SIZE,
                        help=f"Number of teams that break (default: {BREAK_SIZE})")
    parser.add_argument("--rounds-left", type=int, default=ROUNDS_REMAINING,
                        help=f"Rounds remaining (default: {ROUNDS_REMAINING})")
    args = parser.parse_args()

    BREAK_SIZE = args.break_size
    ROUNDS_REMAINING = args.rounds_left

    # Parse seed
    seed = None if args.seed == "random" else int(args.seed)

    # Load data
    if args.csv:
        print(f"Loading standings from CSV: {args.csv}", file=sys.stderr)
        teams = load_csv(args.csv)
    elif args.url:
        print(f"Scraping standings from: {args.url}", file=sys.stderr)
        teams = scrape_tabbycat(args.url)
    else:
        parser.error("Provide either a URL or --csv path")

    print(f"Found {len(teams)} active teams\n", file=sys.stderr)

    # Warn if team count not divisible by 4
    if len(teams) % 4 != 0:
        remainder = len(teams) % 4
        print(f"⚠  {len(teams)} teams is not divisible by 4 "
              f"({remainder} team(s) in an incomplete room each round).\n",
              file=sys.stderr)

    # Run simulations
    print(f"Running {args.sims:,} simulations (seed={args.seed})...",
          file=sys.stderr)
    results = run_simulations(teams, args.sims, seed)

    # Print results
    print_results(teams, results)


if __name__ == "__main__":
    main()

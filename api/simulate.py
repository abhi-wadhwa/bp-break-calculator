"""Vercel serverless function: Monte Carlo break simulation."""

import json
import random
from collections import defaultdict
from http.server import BaseHTTPRequestHandler

BP_POINTS = [3, 2, 1, 0]
MAX_SIMS = 500_000


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            teams = body.get("teams", [])
            rounds_remaining = int(body.get("rounds_remaining", 2))
            break_size = int(body.get("break_size", 16))
            num_sims = min(int(body.get("num_sims", 100_000)), MAX_SIMS)
            seed = body.get("seed")
            novice_teams = set(body.get("novice_teams", []))
            novice_break_size = int(body.get("novice_break_size", 4))

            if not teams:
                return self._json(400, {"error": "No teams provided"})

            results = run_simulations(
                teams, num_sims, rounds_remaining, break_size,
                novice_teams, novice_break_size, seed,
            )
            self._json(200, results)
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def simulate_once(points, num_teams, rounds, rng):
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
                    novice_set, novice_break_size, seed):
    rng = random.Random(seed)
    num_teams = len(teams)
    base_points = [t["points"] for t in teams]
    team_names = [t["team"] for t in teams]
    is_novice = [t["team"] in novice_set for t in teams]

    break_line_dist = defaultdict(int)
    team_break_counts = defaultdict(int)
    team_points_sums = defaultdict(float)
    team_points_min = {n: 999 for n in team_names}
    team_points_max = {n: 0 for n in team_names}

    novice_break_line_dist = defaultdict(int)
    novice_break_counts = defaultdict(int)

    # Track per-breakline: how many teams on that exact point value,
    # and how many of those actually break
    # Key: break_line_pts -> list of (teams_on_line, teams_breaking_from_line)
    break_line_detail = defaultdict(lambda: defaultdict(int))

    for _ in range(num_sims):
        final_pts = simulate_once(base_points, num_teams, rounds_remaining, rng)

        standings = [(i, final_pts[i], rng.random()) for i in range(num_teams)]
        standings.sort(key=lambda x: (-x[1], x[2]))

        if len(standings) >= break_size:
            bl_pts = standings[break_size - 1][1]
            break_line_dist[bl_pts] += 1

            # Count teams on the break line point value
            teams_on_line = sum(1 for _, p, _ in standings if p == bl_pts)
            # Count how many of those break (are in top break_size)
            breaking_from_line = sum(
                1 for rank, (_, p, _) in enumerate(standings)
                if p == bl_pts and rank < break_size
            )
            not_breaking = teams_on_line - breaking_from_line
            break_line_detail[bl_pts][(breaking_from_line, not_breaking)] += 1

        open_breakers = set()
        for rank, (idx, pts, _) in enumerate(standings):
            name = team_names[idx]
            team_points_sums[name] += pts
            if pts < team_points_min[name]:
                team_points_min[name] = pts
            if pts > team_points_max[name]:
                team_points_max[name] = pts
            if rank < break_size:
                team_break_counts[name] += 1
                open_breakers.add(idx)

        if novice_set:
            novice_eligible = []
            for _rank, (idx, pts, _) in enumerate(standings):
                if is_novice[idx] and idx not in open_breakers:
                    novice_eligible.append((idx, pts))
                    if len(novice_eligible) == novice_break_size:
                        break

            if novice_eligible:
                novice_break_line_dist[novice_eligible[-1][1]] += 1

            novice_breaking = {idx for idx, _ in novice_eligible}
            for idx in range(num_teams):
                if is_novice[idx] and (idx in open_breakers or idx in novice_breaking):
                    novice_break_counts[team_names[idx]] += 1

    # Build aggregated response
    team_results = []
    for t in teams:
        name = t["team"]
        break_pct = team_break_counts.get(name, 0) / num_sims * 100
        avg_final = team_points_sums.get(name, 0) / num_sims
        team_results.append({
            "team": name,
            "current_pts": t["points"],
            "break_pct": round(break_pct, 2),
            "avg_final": round(avg_final, 1),
            "min": team_points_min.get(name, 0),
            "max": team_points_max.get(name, 0),
        })
    team_results.sort(key=lambda x: (-x["break_pct"], -x["current_pts"]))

    novice_results = []
    if novice_set:
        for t in teams:
            name = t["team"]
            if name not in novice_set:
                continue
            open_pct = team_break_counts.get(name, 0) / num_sims * 100
            any_pct = novice_break_counts.get(name, 0) / num_sims * 100
            novice_results.append({
                "team": name,
                "current_pts": t["points"],
                "open_break_pct": round(open_pct, 1),
                "novice_break_pct": round(any_pct - open_pct, 1),
                "any_break_pct": round(any_pct, 1),
                "avg_final": round(team_points_sums.get(name, 0) / num_sims, 1),
            })
        novice_results.sort(key=lambda x: (-x["any_break_pct"], -x["current_pts"]))

    # Serialize break_line_detail:
    # { "pts": [ { "breaking": N, "not_breaking": M, "count": C }, ... ] }
    detail_out = {}
    for pts_val in sorted(break_line_detail.keys()):
        scenarios = []
        for (brk, not_brk), count in sorted(
            break_line_detail[pts_val].items(),
            key=lambda x: -x[1],
        ):
            scenarios.append({
                "breaking": brk,
                "not_breaking": not_brk,
                "count": count,
            })
        detail_out[str(pts_val)] = scenarios

    return {
        "break_line_dist": {str(k): v for k, v in sorted(break_line_dist.items())},
        "break_line_detail": detail_out,
        "team_results": team_results,
        "novice_break_line_dist": {str(k): v for k, v in sorted(novice_break_line_dist.items())},
        "novice_results": novice_results,
        "num_sims": num_sims,
    }

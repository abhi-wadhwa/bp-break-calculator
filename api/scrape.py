"""Vercel serverless function: scrape Tabbycat/Calicotab standings."""

import json
import re
import ssl
from http.server import BaseHTTPRequestHandler
from urllib.request import urlopen, Request


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            url = body.get("url", "").strip()

            if not url:
                return self._json(400, {"error": "Missing URL"})

            teams, tournament_name, rounds_completed = scrape_tabbycat(url)
            self._json(200, {
                "teams": teams,
                "tournament_name": tournament_name,
                "rounds_completed": rounds_completed,
            })
        except ValueError as e:
            self._json(400, {"error": str(e)})
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


def scrape_tabbycat(url: str):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=15, context=ctx) as resp:
        html = resp.read().decode("utf-8")

    # Tournament name from <title>
    title_match = re.search(r"<title>(.*?)</title>", html)
    tournament_name = ""
    if title_match:
        raw = title_match.group(1)
        tournament_name = raw.split("|")[0].split("\u2014")[0].strip()
        tournament_name = tournament_name.replace("Tabbycat", "").strip(" |\u2013\u2014")

    # Extract embedded JSON standings
    match = re.search(r"tablesData:\s*(\[.+\])\s*\}", html, re.DOTALL)
    if not match:
        raise ValueError(
            "Could not find standings data. "
            "The page may require JavaScript rendering or has a different format."
        )

    tables_data = json.loads(match.group(1))
    table = tables_data[0]

    round_cols = [h for h in table["head"] if h.get("key", "").startswith("r")]
    num_rounds_shown = len(round_cols)

    teams = []
    for row in table["data"]:
        team_name = row[0].get("text", "Unknown")
        points = int(row[1].get("sort", row[1].get("text", 0)))

        rounds_played = 0
        for i in range(2, len(row)):
            if row[i].get("text", "\u2014") != "\u2014":
                rounds_played += 1

        teams.append({
            "team": team_name,
            "points": points,
            "rounds_played": rounds_played,
        })

    min_rounds = max(1, num_rounds_shown // 2)
    active = [
        t for t in teams
        if t["rounds_played"] >= min_rounds and "Swing" not in t["team"]
    ]
    active.sort(key=lambda t: -t["points"])

    # Strip internal fields before returning
    clean = [{"team": t["team"], "points": t["points"]} for t in active]
    return clean, tournament_name, num_rounds_shown

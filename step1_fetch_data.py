"""
STEP 1: Fetch Chess.com Data
=============================
This script pulls real game data from Chess.com's free public API.
No API key needed. Just run it and it builds your dataset CSV.

HOW TO RUN:
    python step1_fetch_data.py

OUTPUT:
    chess_dataset.csv  (your training data)
"""

import requests
import pandas as pd
import time
import re

# ─────────────────────────────────────────
#  CONFIG — edit these if you want
# ─────────────────────────────────────────
USERNAMES = [
    "hikaru",       # Super GM, aggressive style
    "magnuscarlsen", # World champion, balanced
    "gothamchess",  # Popular streamer, educational
    "BotezLive",    # Popular streamer
    "danielnaroditsky", # Speed chess specialist
    # ADD YOUR OWN Chess.com username below!
    # "your_username_here",
]

MONTHS_TO_FETCH = 3   # How many recent months of games per player
HEADERS = {"User-Agent": "ChessOpeningRecommender/1.0 student-ml-project"}

# ─────────────────────────────────────────
#  OPENING FAMILY GROUPING
#  We group ECO codes into 5 broad styles
# ─────────────────────────────────────────
def get_opening_family(opening_name):
    """Map an opening name to a broad style category."""
    if not opening_name:
        return "Unknown"

    opening_lower = opening_name.lower()

    # Aggressive / Tactical openings
    if any(x in opening_lower for x in [
        "sicilian", "king's gambit", "vienna", "danish", "fried liver",
        "halloween", "latvian", "budapest", "alekhine", "pirc", "modern"
    ]):
        return "Aggressive"

    # Solid / Classical openings
    if any(x in opening_lower for x in [
        "ruy lopez", "italian", "four knights", "three knights",
        "giuoco", "berlin", "marshall", "petrov"
    ]):
        return "Classical"

    # Closed / Strategic openings
    if any(x in opening_lower for x in [
        "queen's gambit", "nimzo", "king's indian", "grunfeld",
        "bogo", "catalan", "english", "reti", "london", "colle"
    ]):
        return "Strategic"

    # Hyper-modern openings
    if any(x in opening_lower for x in [
        "caro-kann", "french", "slav", "dutch", "benoni",
        "benko", "volga", "symmetrical"
    ]):
        return "Solid"

    return "Other"


# ─────────────────────────────────────────
#  PARSE A SINGLE GAME FROM PGN TEXT
# ─────────────────────────────────────────
def parse_game(game_str, username):
    """Extract features from a single PGN game string."""

    def get_tag(tag):
        match = re.search(rf'\[{tag} "(.+?)"\]', game_str)
        return match.group(1) if match else None

    white = get_tag("White") or ""
    black = get_tag("Black") or ""
    result = get_tag("Result")
    white_elo = get_tag("WhiteElo")
    black_elo = get_tag("BlackElo")
    time_class = get_tag("TimeClass")
    opening = get_tag("Opening")
    termination = get_tag("Termination") or ""

    # Skip games with missing info
    if not all([result, white_elo, black_elo, opening]):
        return None

    try:
        white_elo = int(white_elo)
        black_elo = int(black_elo)
    except:
        return None

    # Determine if our player was white or black
    username_lower = username.lower()
    if white.lower() == username_lower:
        player_elo = white_elo
        opponent_elo = black_elo
        played_as = "white"
        if result == "1-0":
            outcome = "win"
        elif result == "0-1":
            outcome = "loss"
        else:
            outcome = "draw"
    elif black.lower() == username_lower:
        player_elo = black_elo
        opponent_elo = white_elo
        played_as = "black"
        if result == "0-1":
            outcome = "win"
        elif result == "1-0":
            outcome = "loss"
        else:
            outcome = "draw"
    else:
        return None

    # Count number of moves (rough estimate from move numbers)
    moves = re.findall(r'\d+\.', game_str)
    num_moves = len(moves)

    # Was it decisive? (not a draw)
    decisive = 1 if outcome in ["win", "loss"] else 0

    # Was it a resign or timeout? (aggressive/tactical games often end in resign)
    resigned = 1 if "resign" in termination.lower() else 0

    return {
        "username": username,
        "player_elo": player_elo,
        "opponent_elo": opponent_elo,
        "elo_diff": player_elo - opponent_elo,
        "played_as": played_as,
        "time_class": time_class or "unknown",
        "opening_name": opening,
        "opening_family": get_opening_family(opening),
        "outcome": outcome,
        "num_moves": num_moves,
        "decisive": decisive,
        "resigned": resigned,
    }


# ─────────────────────────────────────────
#  FETCH GAMES FOR ONE PLAYER
# ─────────────────────────────────────────
def fetch_player_games(username, months=3):
    """Fetch recent games for a Chess.com username."""
    print(f"\n Fetching data for: {username}")

    # Get list of available monthly archives
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    try:
        r = requests.get(archives_url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        archives = r.json().get("archives", [])
    except Exception as e:
        print(f"   Could not fetch archives for {username}: {e}")
        return []

    if not archives:
        print(f"   No archives found for {username}")
        return []

    # Take most recent N months
    recent_archives = archives[-months:]
    all_games = []

    for archive_url in recent_archives:
        month_label = "/".join(archive_url.split("/")[-2:])
        print(f"   Fetching {month_label}...", end=" ")

        try:
            r = requests.get(archive_url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            games_data = r.json().get("games", [])
        except Exception as e:
            print(f"failed ({e})")
            continue

        # Each game has a 'pgn' field with the full game text
        count = 0
        for game in games_data:
            pgn = game.get("pgn", "")
            if not pgn:
                continue
            parsed = parse_game(pgn, username)
            if parsed:
                all_games.append(parsed)
                count += 1

        print(f"{count} games parsed")
        time.sleep(0.5)  # be polite to Chess.com servers

    return all_games


# ─────────────────────────────────────────
#  FETCH PLAYER STATS (win rates etc.)
# ─────────────────────────────────────────
def fetch_player_stats(username):
    """Get overall stats for a player."""
    url = f"https://api.chess.com/pub/player/{username}/stats"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        stats = r.json()
    except:
        return {}

    result = {}
    for time_class in ["chess_rapid", "chess_blitz", "chess_bullet"]:
        if time_class in stats:
            tc_data = stats[time_class]
            record = tc_data.get("record", {})
            wins = record.get("win", 0)
            losses = record.get("loss", 0)
            draws = record.get("draw", 0)
            total = wins + losses + draws
            result[f"{time_class}_rating"] = tc_data.get("last", {}).get("rating", 0)
            result[f"{time_class}_winrate"] = round(wins / total, 3) if total > 0 else 0

    return result


# ─────────────────────────────────────────
#  MAIN — run everything
# ─────────────────────────────────────────
def main():
    all_records = []

    for username in USERNAMES:
        games = fetch_player_games(username, months=MONTHS_TO_FETCH)
        stats = fetch_player_stats(username)

        for game in games:
            game.update(stats)
            all_records.append(game)

    if not all_records:
        print("\n No data collected. Check your internet connection.")
        return

    df = pd.DataFrame(all_records)

    # Drop rows with missing opening family
    df = df[df["opening_family"] != "Unknown"]
    df = df[df["opening_family"] != "Other"]

    print(f"\n Total games collected: {len(df)}")
    print(f"\n Opening family distribution:")
    print(df["opening_family"].value_counts())
    print(f"\n Time class distribution:")
    print(df["time_class"].value_counts())

    # Save to CSV
    df.to_csv("chess_dataset.csv", index=False)
    print(f"\n Saved to chess_dataset.csv")
    print(df.head())


if __name__ == "__main__":
    main()

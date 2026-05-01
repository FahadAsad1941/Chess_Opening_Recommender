"""
STEP 4: Chess Opening Recommender
===================================
The final product. Enter a Chess.com username and get
personalized opening recommendations with explanations.

HOW TO RUN:
    python step4_recommender.py

INPUT:  best_model.pkl  (from step 3)
        Chess.com API  (live fetch for new users)
"""

import requests
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

HEADERS = {"User-Agent": "ChessOpeningRecommender/1.0 student-ml-project"}

# ─────────────────────────────────────────
#  OPENING RECOMMENDATIONS DATABASE
#  Maps opening family → specific openings with details
# ─────────────────────────────────────────
OPENING_DB = {
    "Aggressive": {
        "openings": [
            {
                "name": "Sicilian Defense: Najdorf Variation",
                "moves": "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6",
                "why": "Sharp, unbalanced positions with attacking chances for Black",
                "famous_players": ["Fischer", "Kasparov", "Najdorf"],
                "difficulty": "Advanced",
                "win_rate_white": "52%", "win_rate_black": "48%"
            },
            {
                "name": "King's Gambit",
                "moves": "1.e4 e5 2.f4",
                "why": "Sacrifices a pawn immediately for rapid development and attack",
                "famous_players": ["Tal", "Spassky", "Morphy"],
                "difficulty": "Intermediate",
                "win_rate_white": "55%", "win_rate_black": "45%"
            },
            {
                "name": "Vienna Game: Frankenstein-Dracula Variation",
                "moves": "1.e4 e5 2.Nc3 Nf6 3.Bc4 Nxe4",
                "why": "Extremely sharp and tactical — leads to forced lines",
                "famous_players": ["Spielmann"],
                "difficulty": "Advanced",
                "win_rate_white": "53%", "win_rate_black": "47%"
            },
        ],
        "style_description": "You play aggressively, preferring short decisive games. You like to attack early and put your opponent under pressure immediately.",
        "tip": "Study tactics puzzles daily — your style lives and dies by calculation."
    },
    "Classical": {
        "openings": [
            {
                "name": "Ruy Lopez (Spanish Opening)",
                "moves": "1.e4 e5 2.Nf3 Nc6 3.Bb5",
                "why": "The most classical opening. Puts long-term pressure on e5 pawn.",
                "famous_players": ["Karpov", "Capablanca", "Fischer"],
                "difficulty": "Intermediate",
                "win_rate_white": "54%", "win_rate_black": "46%"
            },
            {
                "name": "Italian Game: Giuoco Piano",
                "moves": "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5",
                "why": "Solid development, good for building classical understanding",
                "famous_players": ["Greco", "Carlsen"],
                "difficulty": "Beginner-Intermediate",
                "win_rate_white": "53%", "win_rate_black": "47%"
            },
            {
                "name": "Petrov's Defense",
                "moves": "1.e4 e5 2.Nf3 Nf6",
                "why": "Ultra-solid symmetrical response, very drawish at high levels",
                "famous_players": ["Petrov", "Kramnik"],
                "difficulty": "Intermediate",
                "win_rate_white": "50%", "win_rate_black": "50%"
            },
        ],
        "style_description": "You favor principled, classical chess. You follow sound opening principles: control center, develop pieces, castle early.",
        "tip": "Study endgames — classical players often win in long technical endgames."
    },
    "Strategic": {
        "openings": [
            {
                "name": "Queen's Gambit Declined",
                "moves": "1.d4 d5 2.c4 e6",
                "why": "Solid structure, strategic pawn play, long-term pressure",
                "famous_players": ["Karpov", "Petrosian", "Carlsen"],
                "difficulty": "Intermediate",
                "win_rate_white": "56%", "win_rate_black": "44%"
            },
            {
                "name": "King's Indian Defense",
                "moves": "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7",
                "why": "Hypermodern — lets White take center, then counterattacks it",
                "famous_players": ["Fischer", "Kasparov", "Bronstein"],
                "difficulty": "Advanced",
                "win_rate_white": "51%", "win_rate_black": "49%"
            },
            {
                "name": "English Opening",
                "moves": "1.c4",
                "why": "Flexible, transposes into many systems. Strategic depth.",
                "famous_players": ["Karpov", "Botvinnik", "Kramnik"],
                "difficulty": "Intermediate-Advanced",
                "win_rate_white": "55%", "win_rate_black": "45%"
            },
        ],
        "style_description": "You are a strategic thinker who prefers closed positions, long-term plans, and positional pressure over sharp tactics.",
        "tip": "Study pawn structures — strategic players win by understanding which pawn breaks to make."
    },
    "Solid": {
        "openings": [
            {
                "name": "Caro-Kann Defense",
                "moves": "1.e4 c6",
                "why": "Rock-solid structure, avoids early complications",
                "famous_players": ["Petrosian", "Karpov", "Short"],
                "difficulty": "Intermediate",
                "win_rate_white": "53%", "win_rate_black": "47%"
            },
            {
                "name": "French Defense",
                "moves": "1.e4 e6",
                "why": "Solid but slightly passive — leads to rich pawn structures",
                "famous_players": ["Nimzowitsch", "Uhlmann"],
                "difficulty": "Intermediate",
                "win_rate_white": "54%", "win_rate_black": "46%"
            },
            {
                "name": "Slav Defense",
                "moves": "1.d4 d5 2.c4 c6",
                "why": "Extremely solid against QG — solid pawn chain with counterplay",
                "famous_players": ["Geller", "Kramnik", "Anand"],
                "difficulty": "Intermediate",
                "win_rate_white": "54%", "win_rate_black": "46%"
            },
        ],
        "style_description": "You value safety and solidity. You avoid early complications and prefer to get a playable position before launching plans.",
        "tip": "Work on converting small advantages — solid players win by slowly squeezing opponents."
    },
}

# ─────────────────────────────────────────
#  FETCH PLAYER FEATURES FROM CHESS.COM
# ─────────────────────────────────────────
def fetch_player_features(username, months=2):
    """Fetch and compute features for a single player from Chess.com."""
    print(f"\n   Fetching live data for '{username}' from Chess.com...")

    # Get stats
    stats_url = f"https://api.chess.com/pub/player/{username}/stats"
    try:
        r = requests.get(stats_url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        stats = r.json()
    except Exception as e:
        print(f"   Could not fetch stats: {e}")
        return None

    # Get recent games for behavioral features
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    try:
        r = requests.get(archives_url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        archives = r.json().get("archives", [])[-months:]
    except:
        archives = []

    # Parse games
    import re
    games_data = []
    for archive_url in archives:
        try:
            r = requests.get(archive_url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            games = r.json().get("games", [])
            for game in games:
                pgn = game.get("pgn", "")
                if not pgn:
                    continue
                white = re.search(r'\[White "(.+?)"\]', pgn)
                result = re.search(r'\[Result "(.+?)"\]', pgn)
                white_elo = re.search(r'\[WhiteElo "(.+?)"\]', pgn)
                black_elo = re.search(r'\[BlackElo "(.+?)"\]', pgn)
                moves_list = re.findall(r'\d+\.', pgn)
                termination = re.search(r'\[Termination "(.+?)"\]', pgn)

                if not all([result, white_elo, black_elo]):
                    continue

                is_white = white and white.group(1).lower() == username.lower()
                try:
                    welo = int(white_elo.group(1))
                    belo = int(black_elo.group(1))
                except:
                    continue

                res = result.group(1)
                if is_white:
                    outcome = "win" if res == "1-0" else ("loss" if res == "0-1" else "draw")
                    player_elo = welo
                    opp_elo = belo
                else:
                    outcome = "win" if res == "0-1" else ("loss" if res == "1-0" else "draw")
                    player_elo = belo
                    opp_elo = welo

                games_data.append({
                    "outcome": outcome,
                    "player_elo": player_elo,
                    "opponent_elo": opp_elo,
                    "num_moves": len(moves_list),
                    "decisive": 1 if outcome != "draw" else 0,
                    "resigned": 1 if termination and "resign" in termination.group(1).lower() else 0,
                    "played_as": "white" if is_white else "black",
                })
        except:
            continue

    if not games_data:
        print("   No recent games found — using stats only")
        games_data = [{"outcome": "win", "player_elo": 1200, "opponent_elo": 1200,
                       "num_moves": 35, "decisive": 1, "resigned": 0, "played_as": "white"}]

    gdf = pd.DataFrame(games_data)

    # Determine most common time class
    time_class_map = {"bullet": 0, "blitz": 1, "rapid": 2, "daily": 3}
    best_tc = "blitz"
    best_rating = 0
    for tc in ["chess_rapid", "chess_blitz", "chess_bullet"]:
        if tc in stats:
            rating = stats[tc].get("last", {}).get("rating", 0)
            if rating > best_rating:
                best_rating = rating
                best_tc = tc.replace("chess_", "")

    features = {
        "player_elo": gdf["player_elo"].mean(),
        "opponent_elo": gdf["opponent_elo"].mean(),
        "elo_diff": (gdf["player_elo"] - gdf["opponent_elo"]).mean(),
        "played_as_enc": (gdf["played_as"] == "white").mean(),
        "time_class_enc": time_class_map.get(best_tc, 1),
        "num_moves": gdf["num_moves"].mean(),
        "decisive": gdf["decisive"].mean(),
        "resigned": gdf["resigned"].mean(),
        "win_rate": (gdf["outcome"] == "win").mean(),
        "draw_rate": (gdf["outcome"] == "draw").mean(),
        "decisive_rate": gdf["decisive"].mean(),
        "resign_rate": gdf["resigned"].mean(),
        "aggression_score": 1 / (gdf["num_moves"].mean() + 1),
    }

    return features, len(games_data), best_rating, best_tc


# ─────────────────────────────────────────
#  VISUALIZE RECOMMENDATION
# ─────────────────────────────────────────
def visualize_recommendation(username, predicted_family, probabilities, 
                               classes, features, model_name):
    """Generate a nice recommendation visualization."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Chess Opening Recommendation for: {username}", 
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    family_colors = {
        "Aggressive": "#D85A30",
        "Classical":  "#378ADD",
        "Strategic":  "#1D9E75",
        "Solid":      "#7F77DD",
    }
    pred_color = family_colors.get(predicted_family, "#888780")

    # ── Panel 1: Predicted style badge ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.add_patch(plt.Rectangle((0.05, 0.3), 0.9, 0.4, 
                                  color=pred_color, alpha=0.15, 
                                  transform=ax1.transAxes, clip_on=False))
    ax1.text(0.5, 0.65, "Your Playing Style", ha="center", va="center",
             fontsize=10, color="gray", transform=ax1.transAxes)
    ax1.text(0.5, 0.5, predicted_family, ha="center", va="center",
             fontsize=20, fontweight="bold", color=pred_color,
             transform=ax1.transAxes)
    ax1.text(0.5, 0.35, f"Predicted by {model_name}", ha="center", va="center",
             fontsize=8, color="gray", transform=ax1.transAxes)

    # ── Panel 2: Probability bars ──
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    names_sorted  = [x[0] for x in sorted_probs]
    probs_sorted  = [x[1] for x in sorted_probs]
    bar_colors    = [family_colors.get(n, "#888780") for n in names_sorted]
    bars = ax2.barh(names_sorted, probs_sorted, color=bar_colors, height=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    ax2.set_title("Style Probabilities")
    for bar, p in zip(bars, probs_sorted):
        ax2.text(p + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{p:.1%}", va="center", fontsize=9)

    # ── Panel 3: Player stats radar ──
    ax3 = fig.add_subplot(gs[0, 2])
    stats_display = {
        "Win Rate":      features.get("win_rate", 0),
        "Aggression":    min(features.get("aggression_score", 0) * 20, 1),
        "Draw Rate":     features.get("draw_rate", 0),
        "Resign Rate":   features.get("resign_rate", 0),
        "Decisive Rate": features.get("decisive_rate", 0),
    }
    stat_names = list(stats_display.keys())
    stat_vals  = list(stats_display.values())
    bars3 = ax3.barh(stat_names, stat_vals,
                     color=[pred_color] * len(stat_names), alpha=0.7, height=0.5)
    ax3.set_xlim(0, 1)
    ax3.set_title("Your Player Profile")
    ax3.set_xlabel("Rate (0–1)")
    for bar, v in zip(bars3, stat_vals):
        ax3.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.1%}", va="center", fontsize=9)

    # ── Bottom: Opening cards ──
    openings = OPENING_DB[predicted_family]["openings"][:3]
    for i, opening in enumerate(openings):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Card background
        card = plt.Rectangle((0, 0), 1, 1, linewidth=1,
                               edgecolor=pred_color, facecolor="white",
                               transform=ax.transAxes)
        ax.add_patch(card)

        # Rank badge
        ax.text(0.08, 0.90, f"#{i+1}", transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=pred_color, va="top")

        # Opening name
        name_text = opening["name"]
        if len(name_text) > 35:
            name_text = name_text[:32] + "..."
        ax.text(0.5, 0.80, name_text, transform=ax.transAxes,
                fontsize=9, fontweight="bold", ha="center", va="top",
                wrap=True)

        # Moves
        ax.text(0.5, 0.67, opening["moves"], transform=ax.transAxes,
                fontsize=8, ha="center", va="top", color="#534AB7",
                fontfamily="monospace")

        # Why
        why_text = opening["why"]
        if len(why_text) > 60:
            why_text = why_text[:57] + "..."
        ax.text(0.5, 0.52, why_text, transform=ax.transAxes,
                fontsize=7.5, ha="center", va="top", color="gray",
                style="italic")

        # Famous players
        players = ", ".join(opening["famous_players"][:2])
        ax.text(0.5, 0.35, f"Famous players: {players}",
                transform=ax.transAxes, fontsize=7.5, ha="center",
                va="top", color="#444441")

        # Difficulty badge
        diff_colors = {"Beginner-Intermediate": "#1D9E75",
                       "Intermediate": "#378ADD",
                       "Intermediate-Advanced": "#EF9F27",
                       "Advanced": "#D85A30"}
        diff_col = diff_colors.get(opening["difficulty"], "#888780")
        ax.text(0.5, 0.20, f"Difficulty: {opening['difficulty']}",
                transform=ax.transAxes, fontsize=8, ha="center",
                va="top", color=diff_col, fontweight="bold")

        # Win rates
        ax.text(0.5, 0.09,
                f"White: {opening['win_rate_white']}  |  Black: {opening['win_rate_black']}",
                transform=ax.transAxes, fontsize=7.5, ha="center",
                va="top", color="gray")

    plt.savefig(f"recommendation_{username}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n   Saved: recommendation_{username}.png")


# ─────────────────────────────────────────
#  MAIN RECOMMENDER FUNCTION
# ─────────────────────────────────────────
def recommend(username):
    """Full pipeline: fetch → predict → recommend → visualize."""

    print(f"\n{'='*55}")
    print(f"  CHESS OPENING RECOMMENDER")
    print(f"  Player: {username}")
    print(f"{'='*55}")

    # Load best model
    with open("best_model.pkl", "rb") as f:
        saved = pickle.load(f)

    model       = saved["model"]
    scaler      = saved["scaler"]
    features    = saved["features"]
    classes     = saved["classes"]
    model_name  = saved["model_name"]

    print(f"\n   Using model: {model_name}")
    print(f"   Model accuracies:")
    for n, r in saved["all_results"].items():
        mark = " <-- best" if n == model_name else ""
        print(f"     {n:20s} Acc: {r['acc']:.3f}  F1: {r['f1']:.3f}{mark}")

    # Fetch player features
    result = fetch_player_features(username)
    if result is None:
        print("   Could not fetch player data. Check username and internet connection.")
        return

    player_features, n_games, rating, time_class = result
    print(f"\n   Analyzed {n_games} recent games")
    print(f"   Best rating: {rating} ({time_class})")

    # Build feature vector in correct order
    X_new = pd.DataFrame([player_features])[features]
    X_new_scaled = scaler.transform(X_new)

    # Predict
    # Use scaled if model needs it (SVM, NB), unscaled otherwise
    needs_scale = any(x in model_name for x in ["SVM", "Naive"])
    X_input = X_new_scaled if needs_scale else X_new

    predicted_family = model.predict(X_input)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_input)[0]
    else:
        probabilities = np.zeros(len(classes))
        probabilities[classes.index(predicted_family)] = 1.0

    # Print results
    print(f"\n{'─'*50}")
    print(f"  YOUR PREDICTED STYLE:  {predicted_family.upper()}")
    print(f"{'─'*50}")
    print(f"\n  {OPENING_DB[predicted_family]['style_description']}")
    print(f"\n  Pro tip: {OPENING_DB[predicted_family]['tip']}")

    print(f"\n  TOP 3 RECOMMENDED OPENINGS:")
    print(f"  {'─'*45}")
    for i, opening in enumerate(OPENING_DB[predicted_family]["openings"][:3]):
        print(f"\n  #{i+1}  {opening['name']}")
        print(f"      Moves: {opening['moves']}")
        print(f"      Why:   {opening['why']}")
        print(f"      Level: {opening['difficulty']}")

    # Visualize
    visualize_recommendation(
        username, predicted_family, probabilities,
        classes, player_features, model_name
    )

    print(f"\n Done! Recommendation saved as recommendation_{username}.png")
    return predicted_family


# ─────────────────────────────────────────
#  RUN IT
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n CHESS OPENING RECOMMENDER")
    print(" Enter a Chess.com username to get personalized opening recommendations.")
    print(" (Press Enter to use 'hikaru' as demo)\n")

    username = input("  Chess.com username: ").strip()
    if not username:
        username = "hikaru"

    recommend(username)

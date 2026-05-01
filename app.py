"""
Chess Opening Recommender — GUI App with Interactive Chessboard
HOW TO RUN:
    1. python -m pip install flask requests pandas numpy scikit-learn
    2. python app.py
    3. Open browser: http://localhost:5000
"""
from flask import Flask, render_template_string, jsonify, request
import requests, pandas as pd, numpy as np
import re, threading, time, pickle, os, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

app = Flask(__name__)
HEADERS = {"User-Agent": "ChessOpeningRecommender/1.0 student-project"}

OPENING_DB = {
    "Aggressive": {
        "emoji":"⚔️","color":"#e85d3a",
        "description":"You play fast, sharp, and tactical. You love to attack and create chaos on the board.",
        "tip":"Study tactics puzzles daily — your style wins or loses on calculation.",
        "openings":[
            {"name":"Sicilian Defense: Najdorf","moves":"e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6","pgn":"1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6","difficulty":"Advanced","players":"Fischer, Kasparov","why":"Sharpest reply to 1.e4 — leads to richly complex attacking positions"},
            {"name":"King's Gambit","moves":"e4 e5 f4","pgn":"1.e4 e5 2.f4","difficulty":"Intermediate","players":"Tal, Morphy","why":"Sacrifices a pawn immediately for a blazing kingside attack"},
            {"name":"Vienna Game","moves":"e4 e5 Nc3 Nf6 Bc4","pgn":"1.e4 e5 2.Nc3 Nf6 3.Bc4","difficulty":"Intermediate","players":"Spielmann, Morozevich","why":"Flexible and aggressive — transposes into many sharp tactical lines"},
        ]
    },
    "Classical": {
        "emoji":"♟️","color":"#3a7bd5",
        "description":"You follow sound principles — develop pieces, control the center, castle early.",
        "tip":"Study endgames — classical players convert small advantages in long games.",
        "openings":[
            {"name":"Ruy Lopez (Spanish Opening)","moves":"e4 e5 Nf3 Nc6 Bb5","pgn":"1.e4 e5 2.Nf3 Nc6 3.Bb5","difficulty":"Intermediate","players":"Karpov, Capablanca, Fischer","why":"The most classical opening — puts long-term pressure on the e5 pawn"},
            {"name":"Italian Game","moves":"e4 e5 Nf3 Nc6 Bc4 Bc5","pgn":"1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5","difficulty":"Beginner","players":"Carlsen, Greco","why":"Perfect for learning classical principles — simple and effective"},
            {"name":"Four Knights Game","moves":"e4 e5 Nf3 Nc6 Nc3 Nf6","pgn":"1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6","difficulty":"Beginner","players":"Rubinstein, Marshall","why":"Symmetrical principled development — great for building opening fundamentals"},
        ]
    },
    "Strategic": {
        "emoji":"🧠","color":"#2ecc71",
        "description":"You think long-term. Pawn structures and slow accumulation of advantages define your game.",
        "tip":"Study pawn structures — strategic players win by knowing which breaks to make.",
        "openings":[
            {"name":"Queen's Gambit Declined","moves":"d4 d5 c4 e6 Nc3 Nf6","pgn":"1.d4 d5 2.c4 e6 3.Nc3 Nf6","difficulty":"Intermediate","players":"Karpov, Kramnik, Carlsen","why":"Solid structure with rich positional play and long-term pressure"},
            {"name":"King's Indian Defense","moves":"d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3","pgn":"1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3","difficulty":"Advanced","players":"Fischer, Kasparov","why":"Lets White take the center, then launches a fierce counterattack"},
            {"name":"English Opening","moves":"c4 e5 Nc3 Nf6 Nf3","pgn":"1.c4 e5 2.Nc3 Nf6 3.Nf3","difficulty":"Intermediate","players":"Karpov, Botvinnik","why":"Flexible hypermodern — delays the center fight for strategic maneuvering"},
        ]
    },
    "Solid": {
        "emoji":"🛡️","color":"#9b59b6",
        "description":"You value safety and reliability. You avoid complications and build from a secure foundation.",
        "tip":"Work on converting small advantages — solid players win by slowly outplaying opponents.",
        "openings":[
            {"name":"Caro-Kann Defense","moves":"e4 c6 d4 d5 Nc3 dxe4 Nxe4","pgn":"1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4","difficulty":"Intermediate","players":"Petrosian, Karpov, Anand","why":"Rock-solid structure — avoids early complications without being passive"},
            {"name":"French Defense","moves":"e4 e6 d4 d5 Nc3 Nf6","pgn":"1.e4 e6 2.d4 d5 3.Nc3 Nf6","difficulty":"Intermediate","players":"Nimzowitsch, Uhlmann","why":"Creates a solid pawn chain and leads to rich strategic battles"},
            {"name":"Slav Defense","moves":"d4 d5 c4 c6 Nf3 Nf6 Nc3","pgn":"1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3","difficulty":"Intermediate","players":"Kramnik, Anand","why":"Extremely reliable against the Queen's Gambit — solid with counterplay"},
        ]
    },
}

def get_family_from_moves(pgn_moves_str):
    """Detect opening family from the actual move sequence (first 10 plies).
    Returns a family string or None if undetermined."""
    if not pgn_moves_str: return None
    clean = re.sub(r'\{[^}]*\}', '', pgn_moves_str)
    clean = re.sub(r'\([^)]*\)', '', clean)
    tokens = re.sub(r'\d+\.+', '', clean).split()
    tokens = [t for t in tokens if t and not t.startswith('$') and t not in ('1-0','0-1','1/2-1/2','*')]
    moves = tokens[:10]
    if not moves: return None
    m0 = moves[0].lower()
    m1 = moves[1].lower() if len(moves) > 1 else ''
    m2 = moves[2].lower() if len(moves) > 2 else ''
    m3 = moves[3].lower() if len(moves) > 3 else ''
    m4 = moves[4].lower() if len(moves) > 4 else ''
    if m0 == 'e4':
        if m1 == 'c5':   return 'Aggressive'
        if m1 == 'e6':   return 'Solid'
        if m1 == 'c6':   return 'Solid'
        if m1 == 'd5':   return 'Solid'
        if m1 in ('nf6','g6','d6'): return 'Aggressive'
        if m1 == 'e5':
            if m2 == 'f4':  return 'Aggressive'
            if m2 == 'nc3': return 'Aggressive'
            if m2 == 'nf3':
                if m3 == 'nf6': return 'Classical'
                if m3 == 'nc6':
                    if m4 == 'bb5': return 'Classical'
                    if m4 == 'bc4': return 'Classical'
                    if m4 in ('nc3','d4'): return 'Classical'
                    return 'Classical'
                return 'Classical'
            return 'Classical'
        return 'Classical'
    if m0 == 'd4':
        if m1 == 'nf6':
            if m2 == 'c4':
                if m3 == 'g6':  return 'Strategic'
                if m3 == 'e6':  return 'Strategic'
                if m3 == 'c5':  return 'Strategic'
                return 'Strategic'
            return 'Strategic'
        if m1 == 'd5':
            if m2 == 'c4':  return 'Strategic'
            return 'Solid'
        if m1 == 'f5':  return 'Aggressive'
        if m1 == 'e6':  return 'Solid'
        return 'Strategic'
    if m0 in ('c4', 'nf3', 'g3'): return 'Strategic'
    if m0 == 'f4': return 'Aggressive'
    return None

def get_opening_family(name):
    if not name: return "Classical"  # default instead of None
    o = name.lower()

    # Aggressive / Tactical
    if any(x in o for x in [
        "sicilian","king's gambit","king gambit","vienna","danish","fried liver",
        "latvian","budapest","alekhine","pirc","modern","smith-morra","morra",
        "king's indian attack","halloween","elephant","cochrane","from's gambit",
        "scotch gambit","goring","evan","evans","blackmar","diemer","dutch",
        "leningrad","staunton","trompowsky","tromp","london system attack",
        "bird","polish","orangutan","sokolsky","kings gambit",
        "najdorf","dragon","accelerated dragon","scheveningen","kan","taimanov",
        "pelikan","sveshnikov","grand prix","alapin","bowdler","wing gambit"
    ]): return "Aggressive"

    # Classical / Open games
    if any(x in o for x in [
        "ruy lopez","italian","four knights","three knights","giuoco","berlin",
        "marshall","petrov","spanish","two knights","bishop's opening","center game",
        "ponziani","philidor","hungarian","kings pawn","king pawn","open game",
        "scotch","max lange","fried liver","göring"
    ]): return "Classical"

    # Strategic / Closed / Hypermodern
    if any(x in o for x in [
        "queen's gambit","queens gambit","nimzo","king's indian","kings indian",
        "grunfeld","grünfeld","bogo","catalan","english","reti","réti","london",
        "colle","zukertort","torre","closed","queen's pawn","queens pawn",
        "benoni","benko","volga","fianchetto","hedgehog",
        "richter","veresov","barry","hypermodern"
    ]): return "Strategic"

    # Solid / Semi-open
    if any(x in o for x in [
        "caro-kann","caro kann","french","slav","semi-slav","semi slav",
        "symmetrical","nimzowitsch","scandinavian","center counter","alekhine",
        "owen","st. george","robatsch","gurgenidze","czech","austrian",
        "classical dutch","stonewall"
    ]): return "Solid"

    # If nothing matches, assign based on first move hint
    if o.startswith("e4") or "open" in o: return "Classical"
    if o.startswith("d4") or o.startswith("c4") or o.startswith("nf3"): return "Strategic"

    return "Classical"  # safe default

def parse_pgn_games(pgn_text, username):
    records=[]
    games=re.split(r'\n(?=\[Event )',pgn_text.strip())
    for game_str in games:
        def tag(t):
            m=re.search(rf'\[{t} "([^"]+)"\]',game_str)
            return m.group(1) if m else None
        white,black=tag("White") or "",tag("Black") or ""
        result=tag("Result"); welo_s,belo_s=tag("WhiteElo"),tag("BlackElo")
        tc=tag("TimeClass") or "blitz"; opening=tag("Opening"); term=tag("Termination") or ""
        if not result: continue
        try:
            welo=int(welo_s) if welo_s and welo_s!="?" else None
            belo=int(belo_s) if belo_s and belo_s!="?" else None
        except: continue
        if welo is None or belo is None: continue
        uname=username.lower()
        if white.lower()==uname:
            player_elo,opp_elo,played_as=welo,belo,"white"
            outcome="win" if result=="1-0" else ("loss" if result=="0-1" else "draw")
        elif black.lower()==uname:
            player_elo,opp_elo,played_as=belo,welo,"black"
            outcome="win" if result=="0-1" else ("loss" if result=="1-0" else "draw")
        else: continue
        moves_text=re.sub(r'^(\[.*?\]\s*)+','',game_str,flags=re.DOTALL).strip()
        moves=re.findall(r'\d+\.(?!\.\.)(?!\s*\.)',game_str)
        family=get_family_from_moves(moves_text) or get_opening_family(opening)
        records.append({"username":username,"player_elo":player_elo,"opponent_elo":opp_elo,
            "elo_diff":player_elo-opp_elo,"played_as_enc":1 if played_as=="white" else 0,
            "time_class":tc,"time_class_enc":{"bullet":0,"blitz":1,"rapid":2,"daily":3}.get(tc,1),
            "opening_name":opening or "","opening_family":family,"outcome":outcome,
            "num_moves":max(len(moves),1),"decisive":1 if outcome!="draw" else 0,
            "resigned":1 if "resign" in term.lower() else 0})
    return records

def fetch_all_games(usernames, months=2, progress_cb=None):
    all_records=[]
    for username in usernames:
        if progress_cb: progress_cb(f"📡 Fetching {username}...")
        try:
            r=requests.get(f"https://api.chess.com/pub/player/{username}/games/archives",headers=HEADERS,timeout=10)
            r.raise_for_status(); archives=r.json().get("archives",[])[-months:]
        except Exception as e:
            if progress_cb: progress_cb(f"  ⚠ Skipping {username}: {e}"); continue
        for arch_url in archives:
            label="/".join(arch_url.split("/")[-2:]); pgn_text=""
            try:
                r=requests.get(arch_url+"/pgn",headers=HEADERS,timeout=20); r.raise_for_status(); pgn_text=r.text
            except:
                try:
                    r=requests.get(arch_url,headers=HEADERS,timeout=15); r.raise_for_status()
                    pgn_text="\n\n".join(g.get("pgn","") for g in r.json().get("games",[]) if g.get("pgn"))
                except: continue
            parsed=parse_pgn_games(pgn_text,username)
            if progress_cb: progress_cb(f"  ✓ {username} {label}: {len(parsed)} games")
            all_records.extend(parsed); time.sleep(0.3)
    return all_records

FEATURES=["player_elo","opponent_elo","elo_diff","played_as_enc","time_class_enc",
          "num_moves","decisive","resigned","win_rate","draw_rate","decisive_rate","resign_rate","aggression_score"]

def train_models(records, progress_cb=None):
    df=pd.DataFrame(records); df=df[df["opening_family"].notna()&(df["opening_family"]!="")].copy()
    if len(df)<30: return None,f"Only {len(df)} labeled games. Need at least 30."
    agg=df.groupby("username").agg(
        win_rate=("outcome",lambda x:(x=="win").mean()),draw_rate=("outcome",lambda x:(x=="draw").mean()),
        decisive_rate=("decisive","mean"),resign_rate=("resigned","mean"),
        aggression_score=("num_moves",lambda x:1/(x.mean()+1))).reset_index()
    df=df.merge(agg,on="username")
    feats=[f for f in FEATURES if f in df.columns]
    X=df[feats].fillna(0); y=df["opening_family"]
    if y.nunique()<2: return None,"Not enough opening variety — most games share the same opening family. Try adding more usernames."
    # Only stratify if every class has enough samples for a split
    min_class_count = y.value_counts().min()
    use_stratify = y if min_class_count >= 5 else None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=use_stratify)
    scaler=StandardScaler(); Xtr_s=scaler.fit_transform(X_train); Xte_s=scaler.transform(X_test)
    mdls={"Random Forest":(RandomForestClassifier(n_estimators=100,random_state=42),False),
          "Decision Tree":(DecisionTreeClassifier(max_depth=6,random_state=42),False),
          "Naive Bayes":(GaussianNB(),True),"SVM RBF":(SVC(kernel="rbf",probability=True,random_state=42),True)}
    results={}; best_acc,best_name,best_model=0,None,None
    for name,(model,scaled) in mdls.items():
        if progress_cb: progress_cb(f"🤖 Training {name}...")
        Xtr=Xtr_s if scaled else X_train; Xte=Xte_s if scaled else X_test
        model.fit(Xtr,y_train); acc=accuracy_score(y_test,model.predict(Xte))
        results[name]=round(acc*100,1)
        if progress_cb: progress_cb(f"  ✓ {name}: {round(acc*100,1)}%")
        if acc>best_acc: best_acc,best_name,best_model=acc,name,model
    rf=mdls["Random Forest"][0]
    importances={f:round(float(v),4) for f,v in zip(feats,rf.feature_importances_)}
    saved={"model":best_model,"model_name":best_name,"scaler":scaler,"features":feats,
           "classes":sorted(y.unique()),"results":results,"importances":importances,
           "total_games":len(df),"family_counts":df["opening_family"].value_counts().to_dict()}
    with open("best_model.pkl","wb") as f: pickle.dump(saved,f)
    if progress_cb: progress_cb(f"✅ Best: {best_name} ({round(best_acc*100,1)}%)")
    return saved,None

def recommend_for_user(username):
    if not os.path.exists("best_model.pkl"): return None,"Model not trained yet."
    with open("best_model.pkl","rb") as f: saved=pickle.load(f)
    records=fetch_all_games([username],months=2)
    if not records: return None,f"No games found for '{username}'."
    df=pd.DataFrame(records)
    agg={"player_elo":df["player_elo"].mean(),"opponent_elo":df["opponent_elo"].mean(),
         "elo_diff":df["elo_diff"].mean(),"played_as_enc":df["played_as_enc"].mean(),
         "time_class_enc":df["time_class_enc"].median(),"num_moves":df["num_moves"].mean(),
         "decisive":df["decisive"].mean(),"resigned":df["resigned"].mean(),
         "win_rate":(df["outcome"]=="win").mean(),"draw_rate":(df["outcome"]=="draw").mean(),
         "decisive_rate":df["decisive"].mean(),"resign_rate":df["resigned"].mean(),
         "aggression_score":1/(df["num_moves"].mean()+1)}
    feats=saved["features"]; X_new=pd.DataFrame([agg])[feats].fillna(0)
    model,model_name=saved["model"],saved["model_name"]
    needs_scale=any(x in model_name for x in ["SVM","Naive"])
    X_input=saved["scaler"].transform(X_new) if needs_scale else X_new
    pred=model.predict(X_input)[0]
    probs={}
    if hasattr(model,"predict_proba"):
        raw=model.predict_proba(X_input)[0]
        probs={c:round(float(p)*100,1) for c,p in zip(saved["classes"],raw)}
    else: probs={pred:100.0}
    info=OPENING_DB[pred]
    return {"username":username,"predicted_family":pred,"probabilities":probs,
            "openings":info["openings"],
            "style_info":{"emoji":info["emoji"],"color":info["color"],"description":info["description"],"tip":info["tip"]},
            "player_stats":{"avg_elo":round(agg["player_elo"]),"win_rate":round(agg["win_rate"]*100,1),
                            "draw_rate":round(agg["draw_rate"]*100,1),"avg_moves":round(agg["num_moves"],1),
                            "resign_rate":round(agg["resign_rate"]*100,1),"games_analyzed":len(df)},
            "model_used":model_name,"model_accuracy":saved["results"].get(model_name,0)},None

training_state={"status":"idle","logs":[],"result":None,"error":None}

def run_training():
    global training_state
    training_state={"status":"training","logs":[],"result":None,"error":None}
    TRAIN_USERS=["hikaru","magnuscarlsen","gothamchess","danielnaroditsky","fabianocaruana","anishgiri","penguingm","levonAronian"]
    def log(msg): training_state["logs"].append(msg)
    log("🔄 Connecting to Chess.com API...")
    records=fetch_all_games(TRAIN_USERS,months=2,progress_cb=log)
    log(f"📦 Total records: {len(records)}")
    if not records: training_state["status"]="error"; training_state["error"]="No games fetched."; return
    saved,err=train_models(records,progress_cb=log)
    if err: training_state["status"]="error"; training_state["error"]=err; return
    training_state["status"]="done"
    training_state["result"]={"total_games":saved["total_games"],"model_results":saved["results"],
                               "best_model":saved["model_name"],"family_counts":saved["family_counts"],"importances":saved["importances"]}

@app.route("/")
def index(): return render_template_string(HTML)
@app.route("/api/train",methods=["POST"])
def api_train():
    if training_state["status"]=="training": return jsonify({"error":"Already training"}),400
    threading.Thread(target=run_training,daemon=True).start(); return jsonify({"ok":True})
@app.route("/api/train/status")
def api_train_status(): return jsonify(training_state)
@app.route("/api/recommend",methods=["POST"])
def api_recommend():
    username=request.json.get("username","").strip()
    if not username: return jsonify({"error":"Enter a username"}),400
    result,err=recommend_for_user(username)
    if err: return jsonify({"error":err}),400
    return jsonify(result)
@app.route("/api/model/status")
def api_model_status():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl","rb") as f: saved=pickle.load(f)
        return jsonify({"trained":True,"results":saved["results"],"best":saved["model_name"],"games":saved["total_games"]})
    return jsonify({"trained":False})

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Chess Opening Recommender</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d0d0f;--s1:#141416;--s2:#1c1c20;--border:rgba(255,255,255,0.07);--text:#f0ede8;--muted:#7a7873;--gold:#c9a84c;--gold2:#e8c97a;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;z-index:0;opacity:0.02;background-image:repeating-conic-gradient(#fff 0% 25%,transparent 0% 50%);background-size:48px 48px;pointer-events:none;}
.wrap{position:relative;z-index:1;max-width:1000px;margin:0 auto;padding:0 24px 100px;}
header{padding:48px 0 28px;text-align:center;border-bottom:1px solid var(--border);margin-bottom:36px;}
.logo{font-family:'Playfair Display',serif;font-size:44px;font-weight:900;letter-spacing:-1px;}
.logo span{color:var(--gold);}
.tagline{color:var(--muted);font-size:13px;margin-top:8px;font-family:'DM Mono',monospace;letter-spacing:2px;text-transform:uppercase;}
.card{background:var(--s1);border:1px solid var(--border);border-radius:16px;padding:28px;margin-bottom:20px;}
.card-title{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;margin-bottom:4px;}
.card-sub{color:var(--muted);font-size:13px;margin-bottom:20px;}
.btn{display:inline-flex;align-items:center;gap:8px;padding:11px 22px;border-radius:10px;border:none;font-size:14px;font-weight:500;cursor:pointer;transition:all 0.2s;font-family:'DM Sans',sans-serif;}
.btn-gold{background:var(--gold);color:#000;}
.btn-gold:hover{background:var(--gold2);transform:translateY(-1px);}
.btn:disabled{opacity:0.4;cursor:not-allowed;transform:none!important;}
.input-row{display:flex;gap:10px;}
input[type=text]{flex:1;background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:11px 16px;color:var(--text);font-size:14px;font-family:'DM Mono',monospace;outline:none;transition:border-color 0.2s;}
input[type=text]:focus{border-color:var(--gold);}
input[type=text]::placeholder{color:var(--muted);}
.sbar{display:flex;align-items:center;gap:10px;padding:10px 16px;border-radius:10px;font-size:13px;margin-bottom:16px;}
.sbar.ok{background:rgba(46,204,113,0.1);border:1px solid rgba(46,204,113,0.2);color:#2ecc71;}
.sbar.warn{background:rgba(201,168,76,0.1);border:1px solid rgba(201,168,76,0.2);color:var(--gold);}
.dot{width:8px;height:8px;border-radius:50%;background:currentColor;flex-shrink:0;}
.dot.pulse{animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.logbox{background:#0a0a0c;border:1px solid var(--border);border-radius:10px;padding:16px;font-family:'DM Mono',monospace;font-size:12px;color:#888;height:160px;overflow-y:auto;line-height:1.8;}
.logbox .ok{color:#2ecc71;}
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;}
.mc{background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;display:flex;justify-content:space-between;align-items:center;}
.mc.best{border-color:var(--gold);}
.mcn{font-size:13px;color:var(--muted);}
.mca{font-family:'DM Mono',monospace;font-size:18px;font-weight:500;}
.bbadge{font-size:10px;color:var(--gold);background:rgba(201,168,76,0.15);padding:2px 8px;border-radius:20px;margin-top:4px;display:inline-block;}
#result{display:none;}
.hero{border-radius:16px;padding:32px;margin-bottom:20px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;inset:0;filter:brightness(0.25);z-index:0;background:inherit;}
.hero>*{position:relative;z-index:1;}
.hemoji{font-size:44px;margin-bottom:8px;}
.hname{font-family:'Playfair Display',serif;font-size:34px;font-weight:900;}
.hdesc{color:rgba(255,255,255,0.75);font-size:15px;margin-top:8px;max-width:600px;line-height:1.6;}
.htip{margin-top:14px;font-size:13px;color:rgba(255,255,255,0.55);font-family:'DM Mono',monospace;}
.srow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;}
.sb{background:var(--s1);border:1px solid var(--border);border-radius:12px;padding:16px;text-align:center;}
.sv{font-family:'Playfair Display',serif;font-size:28px;font-weight:700;}
.sl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
.prow{display:flex;flex-direction:column;gap:10px;}
.pi{display:flex;align-items:center;gap:12px;}
.pl{font-size:13px;width:90px;flex-shrink:0;}
.pbw{flex:1;background:var(--s2);border-radius:4px;height:8px;overflow:hidden;}
.pb{height:100%;border-radius:4px;transition:width 0.8s cubic-bezier(.4,0,.2,1);}
.pp{font-family:'DM Mono',monospace;font-size:12px;color:var(--muted);width:40px;text-align:right;}
.stitle{font-family:'Playfair Display',serif;font-size:18px;font-weight:700;margin-bottom:16px;}
/* Opening cards */
.olist{display:flex;flex-direction:column;gap:20px;}
.ocard{background:var(--s2);border:1px solid var(--border);border-radius:14px;overflow:hidden;display:grid;grid-template-columns:1fr 300px;}
.oinfo{padding:24px;}
.orank{font-family:'DM Mono',monospace;font-size:11px;color:var(--gold);margin-bottom:8px;}
.oname{font-family:'Playfair Display',serif;font-size:18px;font-weight:700;line-height:1.3;margin-bottom:10px;}
.opgn{font-family:'DM Mono',monospace;font-size:11px;color:var(--gold);background:rgba(201,168,76,0.08);padding:6px 10px;border-radius:6px;margin-bottom:12px;word-break:break-all;}
.owhy{font-size:13px;color:var(--muted);line-height:1.6;margin-bottom:12px;}
.ometa{font-size:12px;color:var(--muted);border-top:1px solid var(--border);padding-top:12px;}
.ometa strong{color:var(--text);}
/* Board area */
.bwrap{background:#181412;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px;gap:10px;border-left:1px solid var(--border);}
.bcontrols{display:flex;gap:6px;flex-wrap:wrap;justify-content:center;}
.bbtn{background:rgba(255,255,255,0.07);border:none;color:var(--text);border-radius:6px;padding:5px 11px;font-size:12px;cursor:pointer;transition:background 0.15s;font-family:'DM Mono',monospace;}
.bbtn:hover{background:rgba(255,255,255,0.15);}
.mctr{font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);text-align:center;min-height:16px;}
/* Chess board */
.cboard{display:grid;grid-template-columns:repeat(8,34px);grid-template-rows:repeat(8,34px);border:2px solid rgba(255,255,255,0.2);border-radius:3px;overflow:hidden;}
.sq{width:34px;height:34px;display:flex;align-items:center;justify-content:center;font-size:20px;line-height:1;position:relative;}
.sq.light{background:#f0d9b5;}
.sq.dark{background:#b58863;}
.sq .wp{color:#fff;text-shadow:0 0 2px #000,0 1px 3px rgba(0,0,0,0.8);}
.sq .bp{color:#1a1a1a;text-shadow:0 1px 2px rgba(255,255,255,0.5);}
.sq.lm{background-color:rgba(255,213,0,0.4)!important;}
/* rank/file labels */
.board-outer{position:relative;}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(0,0,0,0.3);border-top-color:#000;border-radius:50%;animation:spin 0.7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
#rec-error{color:#e85d3a;font-size:13px;margin-top:10px;display:none;}
#mcredit{text-align:center;margin-top:16px;color:var(--muted);font-size:12px;font-family:'DM Mono',monospace;}
@media(max-width:700px){.ocard{grid-template-columns:1fr;}.mgrid{grid-template-columns:1fr;}.srow{grid-template-columns:1fr 1fr;}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="logo">Chess <span>Opening</span> Recommender</div>
  <div class="tagline">ML-Powered · Chess.com API · Interactive Board</div>
</header>

<div class="card">
  <div class="card-title">Step 1 — Train the Model</div>
  <div class="card-sub">Fetches real games from Chess.com's free public API and trains 4 ML models (Decision Tree, Random Forest, Naive Bayes, SVM). Takes ~2 minutes.</div>
  <div id="sbar" class="sbar warn"><div class="dot" id="sdot"></div><span id="stext">Model not trained yet</span></div>
  <button class="btn btn-gold" id="train-btn" onclick="startTraining()">Train Model</button>
  <div id="tprog" style="display:none;margin-top:16px;"><div class="logbox" id="logbox"></div></div>
  <div id="mresults" style="display:none;margin-top:20px;">
    <div class="stitle" style="margin-bottom:10px;">Model Comparison</div>
    <div class="mgrid" id="mgrid"></div>
  </div>
</div>

<div class="card">
  <div class="card-title">Step 2 — Get Your Recommendation</div>
  <div class="card-sub">Enter any Chess.com username — we'll analyze their games, predict their style, and show matched openings on an interactive chessboard.</div>
  <div class="input-row">
    <input type="text" id="uinput" placeholder="e.g. hikaru, magnuscarlsen, your_username" onkeydown="if(event.key==='Enter')getRec()"/>
    <button class="btn btn-gold" id="rec-btn" onclick="getRec()">Analyze →</button>
  </div>
  <div id="rec-error"></div>
</div>

<div id="result">
  <div class="hero" id="hero"><div class="hemoji" id="hemoji"></div><div class="hname" id="hname"></div><div class="hdesc" id="hdesc"></div><div class="htip" id="htip"></div></div>
  <div class="srow" id="srow"></div>
  <div class="card"><div class="stitle">Style Probabilities</div><div class="prow" id="prow"></div></div>
  <div class="card"><div class="stitle">Top 3 Recommended Openings — with Interactive Board</div><div class="olist" id="olist"></div></div>
  <div id="mcredit"></div>
</div>
</div>

<script>
// ═══════════════════════════════════
//  CHESS ENGINE
// ═══════════════════════════════════
const PC={wK:'♔',wQ:'♕',wR:'♖',wB:'♗',wN:'♘',wP:'♙',bK:'♚',bQ:'♛',bR:'♜',bB:'♝',bN:'♞',bP:'♟'};
const FILES='abcdefgh';

function initBoard(){
  return [
    ['bR','bN','bB','bQ','bK','bB','bN','bR'],
    ['bP','bP','bP','bP','bP','bP','bP','bP'],
    [null,null,null,null,null,null,null,null],
    [null,null,null,null,null,null,null,null],
    [null,null,null,null,null,null,null,null],
    [null,null,null,null,null,null,null,null],
    ['wP','wP','wP','wP','wP','wP','wP','wP'],
    ['wR','wN','wB','wQ','wK','wB','wN','wR'],
  ];
}

function pathClear(b,fr,fc,tr,tc){
  const sr=Math.sign(tr-fr),sc=Math.sign(tc-fc);
  let r=fr+sr,c=fc+sc;
  while(r!==tr||c!==tc){if(b[r][c])return false;r+=sr;c+=sc;}
  return true;
}

function canReach(b,fr,fc,tr,tc,piece,color){
  const dr=tr-fr,dc=tc-fc;
  if(b[tr][tc]&&b[tr][tc][0]===color)return false;
  if(piece==='P'){
    const dir=color==='w'?-1:1;
    if(dc===0){
      if(dr===dir&&!b[tr][tc])return true;
      if(dr===2*dir&&!b[tr][tc]&&!b[fr+dir][fc]&&(color==='w'?fr===6:fr===1))return true;
    }else if(Math.abs(dc)===1&&dr===dir&&b[tr][tc])return true;
    return false;
  }
  if(piece==='N')return(Math.abs(dr)===2&&Math.abs(dc)===1)||(Math.abs(dr)===1&&Math.abs(dc)===2);
  if(piece==='K')return Math.abs(dr)<=1&&Math.abs(dc)<=1;
  if(piece==='R'){if(dr!==0&&dc!==0)return false;return pathClear(b,fr,fc,tr,tc);}
  if(piece==='B'){if(Math.abs(dr)!==Math.abs(dc))return false;return pathClear(b,fr,fc,tr,tc);}
  if(piece==='Q'){if(dr===0||dc===0||Math.abs(dr)===Math.abs(dc))return pathClear(b,fr,fc,tr,tc);return false;}
  return false;
}

function applyMove(board,moveStr,isWhite){
  const b=board.map(r=>[...r]);
  const color=isWhite?'w':'b';
  moveStr=moveStr.replace(/[+#!=?]/g,'').trim();
  if(moveStr==='O-O'||moveStr==='0-0'){const row=isWhite?7:0;b[row][6]=b[row][4];b[row][4]=null;b[row][5]=b[row][7];b[row][7]=null;return b;}
  if(moveStr==='O-O-O'||moveStr==='0-0-0'){const row=isWhite?7:0;b[row][2]=b[row][4];b[row][4]=null;b[row][3]=b[row][0];b[row][0]=null;return b;}
  const toFile=moveStr[moveStr.length-2],toRank=moveStr[moveStr.length-1];
  const toCol=FILES.indexOf(toFile),toRow=8-parseInt(toRank);
  if(toCol<0||toRow<0||toRow>7)return b;
  let pieceType='P';
  if(moveStr[0]>='A'&&moveStr[0]<='Z')pieceType=moveStr[0];
  let hintFile='',hintRank='';
  const inner=moveStr.slice(pieceType==='P'?0:1).replace(/x/,'').slice(0,-2);
  for(const ch of inner){if(ch>='a'&&ch<='h')hintFile=ch;else if(ch>='1'&&ch<='8')hintRank=ch;}
  let fromRow=-1,fromCol=-1;
  outer:for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    const sq=b[r][c];
    if(!sq||sq[0]!==color||sq[1]!==pieceType)continue;
    if(hintFile&&FILES[c]!==hintFile)continue;
    if(hintRank&&(8-r).toString()!==hintRank)continue;
    if(canReach(b,r,c,toRow,toCol,pieceType,color)){fromRow=r;fromCol=c;break outer;}
  }
  if(fromRow<0)return b;
  b[toRow][toCol]=b[fromRow][fromCol];b[fromRow][fromCol]=null;
  if(pieceType==='P'&&(toRow===0||toRow===7))b[toRow][toCol]=color+'Q';
  return b;
}

function parseMoves(s){return s.trim().split(/\s+/).filter(m=>m&&!/^\d+\./.test(m));}

const BS={};

function buildBoardHTML(id){
  let h=`<div class="cboard" id="${id}">`;
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    const light=(r+c)%2===0;
    h+=`<div class="sq ${light?'light':'dark'}" id="${id}_${r}_${c}"></div>`;
  }
  return h+'</div>';
}

function renderBoard(id,board,lmFrom,lmTo){
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    const el=document.getElementById(`${id}_${r}_${c}`);
    if(!el)continue;
    const p=board[r][c];
    if(p&&PC[p]){
      const cls=p[0]==='w'?'wp':'bp';
      el.innerHTML=`<span class="${cls}">${PC[p]}</span>`;
    } else {
      el.innerHTML='';
    }
    el.classList.remove('lm');
    if(lmFrom&&lmFrom[0]===r&&lmFrom[1]===c)el.classList.add('lm');
    if(lmTo&&lmTo[0]===r&&lmTo[1]===c)el.classList.add('lm');
  }
}

function initOpeningBoard(id,movesStr){
  const moves=parseMoves(movesStr);
  const boards=[initBoard()];
  let cur=initBoard();
  for(let i=0;i<moves.length;i++){cur=applyMove(cur,moves[i],i%2===0);boards.push(cur.map(r=>[...r]));}
  BS[id]={moves,currentMove:0,boards};
  renderBoard(id,initBoard(),null,null);
  updateCtr(id);
}

function stepMove(id,dir){
  const s=BS[id];if(!s)return;
  s.currentMove=Math.max(0,Math.min(s.boards.length-1,s.currentMove+dir));
  const lmFrom=s.currentMove>0?null:null;
  renderBoard(id,s.boards[s.currentMove],null,null);
  updateCtr(id);
}

function resetBoard(id){const s=BS[id];if(!s)return;s.currentMove=0;renderBoard(id,s.boards[0],null,null);updateCtr(id);}

function playAll(id){
  const s=BS[id];if(!s)return;
  s.currentMove=0;renderBoard(id,s.boards[0],null,null);updateCtr(id);
  const iv=setInterval(()=>{
    if(s.currentMove>=s.boards.length-1){clearInterval(iv);return;}
    s.currentMove++;renderBoard(id,s.boards[s.currentMove],null,null);updateCtr(id);
  },650);
}

function updateCtr(id){
  const s=BS[id];const el=document.getElementById(`${id}_ctr`);if(!el||!s)return;
  const cur=s.currentMove,total=s.moves.length;
  if(cur===0){el.textContent=`Starting position · ${total} moves total`;}
  else{
    const side=cur%2===1?'White':'Black';
    el.textContent=`Move ${cur}/${total} · ${side}: ${s.moves[cur-1]}`;
  }
}

// ═══════════════════════════════════
//  UI + API
// ═══════════════════════════════════
const COLORS={Aggressive:"#e85d3a",Classical:"#3a7bd5",Strategic:"#2ecc71",Solid:"#9b59b6"};

window.onload=async()=>{
  const r=await fetch("/api/model/status").then(x=>x.json());
  if(r.trained)showMR(r.results,r.best,r.games);
};

async function startTraining(){
  const btn=document.getElementById("train-btn");
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Training...';
  document.getElementById("tprog").style.display="block";
  document.getElementById("sbar").className="sbar warn";
  document.getElementById("stext").textContent="Training in progress...";
  document.getElementById("sdot").classList.add("pulse");
  await fetch("/api/train",{method:"POST"});
  pollTraining();
}

function pollTraining(){
  const lb=document.getElementById("logbox");
  const iv=setInterval(async()=>{
    const st=await fetch("/api/train/status").then(x=>x.json());
    lb.innerHTML=st.logs.map(l=>`<p class="${l.startsWith('✅')?'ok':''}">${l}</p>`).join("");
    lb.scrollTop=lb.scrollHeight;
    if(st.status==="done"){clearInterval(iv);showMR(st.result.model_results,st.result.best_model,st.result.total_games);document.getElementById("train-btn").innerHTML="Re-train";document.getElementById("train-btn").disabled=false;}
    if(st.status==="error"){clearInterval(iv);lb.innerHTML+=`<p style="color:#e85d3a">❌ ${st.error}</p>`;document.getElementById("train-btn").innerHTML="Retry";document.getElementById("train-btn").disabled=false;}
  },1500);
}

function showMR(results,best,totalGames){
  document.getElementById("sbar").className="sbar ok";
  document.getElementById("sdot").classList.remove("pulse");
  document.getElementById("stext").textContent=`✓ Ready · ${totalGames} games · Best model: ${best}`;
  document.getElementById("mgrid").innerHTML=Object.entries(results).sort((a,b)=>b[1]-a[1]).map(([n,acc])=>`
    <div class="mc ${n===best?'best':''}">
      <div><div class="mcn">${n}</div>${n===best?'<div class="bbadge">⭐ Best</div>':''}</div>
      <div class="mca" style="color:${acc>=70?'#2ecc71':acc>=55?'#c9a84c':'#e85d3a'}">${acc}%</div>
    </div>`).join("");
  document.getElementById("mresults").style.display="block";
}

async function getRec(){
  const username=document.getElementById("uinput").value.trim();if(!username)return;
  const btn=document.getElementById("rec-btn");const err=document.getElementById("rec-error");
  btn.disabled=true;btn.innerHTML='<span class="spinner" style="border-color:rgba(0,0,0,0.3);border-top-color:#000"></span> Analyzing...';
  err.style.display="none";document.getElementById("result").style.display="none";
  const r=await fetch("/api/recommend",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username})});
  const data=await r.json();
  btn.disabled=false;btn.innerHTML="Analyze →";
  if(data.error){err.textContent=data.error;err.style.display="block";return;}
  renderResult(data);
}

function renderResult(d){
  const col=COLORS[d.predicted_family]||"#c9a84c";
  const hero=document.getElementById("hero");
  hero.style.background=col;hero.style.border=`1px solid ${col}`;
  document.getElementById("hemoji").textContent=d.style_info.emoji;
  document.getElementById("hname").textContent=d.predicted_family+" Player";
  document.getElementById("hdesc").textContent=d.style_info.description;
  document.getElementById("htip").textContent="💡 "+d.style_info.tip;
  const s=d.player_stats;
  document.getElementById("srow").innerHTML=`
    <div class="sb"><div class="sv" style="color:${col}">${s.avg_elo}</div><div class="sl">Avg ELO</div></div>
    <div class="sb"><div class="sv" style="color:${col}">${s.win_rate}%</div><div class="sl">Win Rate</div></div>
    <div class="sb"><div class="sv" style="color:${col}">${s.games_analyzed}</div><div class="sl">Games Analyzed</div></div>`;
  document.getElementById("prow").innerHTML=Object.entries(d.probabilities).sort((a,b)=>b[1]-a[1]).map(([fam,pct])=>`
    <div class="pi">
      <div class="pl" style="color:${COLORS[fam]||'#888'}">${fam}</div>
      <div class="pbw"><div class="pb" style="width:${pct}%;background:${COLORS[fam]||'#888'}"></div></div>
      <div class="pp">${pct}%</div>
    </div>`).join("");

  // Opening cards with boards
  document.getElementById("olist").innerHTML=d.openings.map((o,i)=>{
    const bid=`board_${i}`;
    return `
    <div class="ocard" style="border-top:3px solid ${col}">
      <div class="oinfo">
        <div class="orank">#${i+1} RECOMMENDATION</div>
        <div class="oname">${o.name}</div>
        <div class="opgn">${o.pgn}</div>
        <div class="owhy">${o.why}</div>
        <div class="ometa"><strong>Difficulty:</strong> ${o.difficulty} &nbsp;·&nbsp; <strong>Famous players:</strong> ${o.players}</div>
      </div>
      <div class="bwrap">
        ${buildBoardHTML(bid)}
        <div class="mctr" id="${bid}_ctr"></div>
        <div class="bcontrols">
          <button class="bbtn" onclick="resetBoard('${bid}')">⏮ Start</button>
          <button class="bbtn" onclick="stepMove('${bid}',-1)">◀</button>
          <button class="bbtn" onclick="playAll('${bid}')">▶ Play</button>
          <button class="bbtn" onclick="stepMove('${bid}',1)">▶|</button>
        </div>
      </div>
    </div>`;
  }).join("");

  setTimeout(()=>d.openings.forEach((o,i)=>initOpeningBoard(`board_${i}`,o.moves)),80);
  document.getElementById("mcredit").textContent=`Predicted by ${d.model_used} · ${d.model_accuracy}% accuracy`;
  document.getElementById("result").style.display="block";
  document.getElementById("result").scrollIntoView({behavior:"smooth"});
}
</script>
</body>
</html>"""

if __name__=="__main__":
    print("\n"+"="*50)
    print("  Chess Opening Recommender — GUI")
    print("="*50)
    print("\n  Open your browser at:  http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print("="*50+"\n")
    port = int(os.environ.get("PORT", 8000))
app.run(debug=False, host="0.0.0.0", port=port)
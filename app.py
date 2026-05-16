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
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
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
            {"name":"Sicilian Defense: Najdorf","moves":"e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6","pgn":"1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6","difficulty":"Advanced","players":"Fischer, Kasparov, Hikaru","why":"Sharpest reply to 1.e4 — leads to richly complex attacking positions","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Sicilian-Defense-Najdorf-Variation",
             "puzzles":[
               {"fen":"r1bqkb1r/1p2pppp/p1np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 7","prompt":"White to move: find the best aggressive continuation after a6","answer":"f4","hint":"Push a pawn to restrict Black and prepare f5"},
               {"fen":"r1bqkb1r/1p2pppp/p1np1n2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R b KQkq - 1 7","prompt":"Black to move: how should Black challenge the Bg5 pin?","answer":"e6","hint":"Release the pin and fight for the center"},
               {"fen":"r1bqk2r/pp2bppp/2nppn2/8/2BNP3/2N5/PPP2PPP/R1BQK2R w KQkq - 4 8","prompt":"White to move: launch the kingside attack","answer":"Bg5","hint":"Pin the knight and increase pressure"},
             ]},
            {"name":"King's Gambit","moves":"e4 e5 f4 exf4 Nf3","pgn":"1.e4 e5 2.f4 exf4 3.Nf3","difficulty":"Intermediate","players":"Tal, Morphy, Spassky","why":"Sacrifices a pawn immediately for a blazing kingside attack","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Kings-Gambit-Accepted",
             "puzzles":[
               {"fen":"rnbqkbnr/pppp1ppp/8/8/4Pp2/5N2/PPPP2PP/RNBQKB1R b KQkq - 1 3","prompt":"Black to move: how do you hold the extra pawn?","answer":"g5","hint":"Protect the f4 pawn aggressively"},
               {"fen":"rnbqkbnr/pppp1p1p/8/6p1/4Pp2/5N2/PPPP2PP/RNBQKB1R w KQkq g6 0 4","prompt":"White to move: exploit Black's weakened kingside","answer":"h4","hint":"Attack the g5 pawn immediately"},
               {"fen":"rnbqk1nr/pppp1pbp/6p1/8/2B1PpP1/5N2/PPPP3P/RNBQK2R b KQkq - 1 6","prompt":"Black to move: best defensive resource","answer":"h6","hint":"Prevent Ng5 threats"},
             ]},
            {"name":"Vienna Game","moves":"e4 e5 Nc3 Nf6 Bc4","pgn":"1.e4 e5 2.Nc3 Nf6 3.Bc4","difficulty":"Intermediate","players":"Spielmann, Morozevich","why":"Flexible and aggressive — transposes into many sharp tactical lines","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Vienna-Game",
             "puzzles":[
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 4 4","prompt":"White to move: what is the most aggressive plan?","answer":"d3","hint":"Solidify the center and prepare Bg5"},
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4","prompt":"Black to move: how to fight for equality?","answer":"Bc5","hint":"Develop and mirror White's bishop"},
               {"fen":"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 2 5","prompt":"White to move: start the attack","answer":"Ng5","hint":"Threaten f7 immediately"},
             ]},
            {"name":"Sicilian Defense: Dragon","moves":"e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6","pgn":"1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 g6","difficulty":"Advanced","players":"Karjakin, Topalov","why":"Double-edged race — you attack the kingside while White goes queenside","elo_min":800,
             "chesscom":"https://www.chess.com/openings/Sicilian-Defense-Dragon-Variation",
             "puzzles":[
               {"fen":"r1bqkb1r/pp2pp1p/2np1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 7","prompt":"White to move: set up the Yugoslav Attack","answer":"Be3","hint":"Prepare queenside castling and a kingside pawn storm"},
               {"fen":"r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQ - 4 9","prompt":"White to move: launch the assault","answer":"f4","hint":"Start the pawn storm before Black attacks on the queenside"},
               {"fen":"r1bq1rk1/pp3pbp/2np1np1/4p3/3NP3/2N1B3/PPPQBPPP/R4RK1 b - - 1 11","prompt":"Black to move: counterattack on the queenside","answer":"a5","hint":"Advance the queenside pawns to create counterplay"},
             ]},
            {"name":"Latvian Gambit","moves":"e4 e5 Nf3 f5","pgn":"1.e4 e5 2.Nf3 f5","difficulty":"Intermediate","players":"Shirov (fan)","why":"Wild and chaotic — perfect for players who love to surprise opponents early","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Latvian-Gambit",
             "puzzles":[
               {"fen":"rnbqkbnr/pppp2pp/8/4pp2/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq f6 0 3","prompt":"White to move: how to best challenge Black's gambit?","answer":"Nxe5","hint":"Take the central pawn immediately"},
               {"fen":"rnbqkbnr/pppp2pp/8/4pN2/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 3","prompt":"Black to move: maintain the attack","answer":"Qf6","hint":"Attack the knight and support the f5 pawn"},
               {"fen":"rnb1kbnr/pppp2pp/5q2/4pN2/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4","prompt":"White to move: defend and keep the advantage","answer":"d4","hint":"Open the center and challenge Black"},
             ]},
            {"name":"Alekhine's Defense","moves":"e4 Nf6 e5 Nd5 d4 d6","pgn":"1.e4 Nf6 2.e5 Nd5 3.d4 d6","difficulty":"Intermediate","players":"Alekhine, Bagirov","why":"Provokes White's pawns to overextend, then attacks them as weaknesses","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Alekhines-Defense",
             "puzzles":[
               {"fen":"rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 4","prompt":"White to move: how to support the e5 pawn?","answer":"c4","hint":"Chase the knight and build a big pawn center"},
               {"fen":"rnbqkb1r/ppp1pppp/3p4/4P3/2Pn4/8/PP3PPP/RNBQKBNR w KQkq - 1 5","prompt":"White to move: deal with the knight on d4","answer":"Be3","hint":"Attack the knight with a developing move"},
               {"fen":"rnbqkb1r/pp2pppp/3p4/4P3/2p5/4B3/PP3PPP/RN1QKBNR b KQkq - 1 6","prompt":"Black to move: undermine White's center","answer":"dxe5","hint":"Strike the overextended pawns"},
             ]},
            {"name":"Budapest Gambit","moves":"d4 Nf6 c4 e5 dxe5 Ng4","pgn":"1.d4 Nf6 2.c4 e5 3.dxe5 Ng4","difficulty":"Intermediate","players":"Rubinstein, Spielmann","why":"An early gambit that leads to wild positions — great shock value in blitz","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Budapest-Gambit",
             "puzzles":[
               {"fen":"rnbqkb1r/pppp1ppp/8/4P3/2P3n1/8/PP2PPPP/RNBQKBNR w KQkq - 1 4","prompt":"White to move: how to hold the extra pawn?","answer":"Nf3","hint":"Defend e5 with a developing move"},
               {"fen":"rnbqkb1r/pppp1ppp/8/4P3/2P3n1/5N2/PP2PPPP/RNBQKB1R b KQkq - 2 4","prompt":"Black to move: regain the pawn","answer":"Bc5","hint":"Develop and attack f2"},
               {"fen":"rnbqk2r/pppp1ppp/8/2b1P3/2P3n1/5N2/PP2PPPP/RNBQKB1R w KQkq - 3 5","prompt":"White to move: defend against Nxf2","answer":"e3","hint":"Shore up the f2 pawn and kick the bishop"},
             ]},
            {"name":"Scotch Gambit","moves":"e4 e5 Nf3 Nc6 d4 exd4 Bc4","pgn":"1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Bc4","difficulty":"Intermediate","players":"Morphy, Kasparov","why":"Fast development with piece activity — leads to open tactical battles","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Scotch-Gambit",
             "puzzles":[
               {"fen":"r1bqkbnr/pppp1ppp/2n5/8/2BpP3/5N2/PPP2PPP/RNBQK2R b KQkq - 1 4","prompt":"Black to move: the critical decision","answer":"Bc5","hint":"Develop the bishop and put pressure on f2"},
               {"fen":"r1bqk1nr/pppp1ppp/2n5/2b5/2BpP3/5N2/PPP2PPP/RNBQ1RK1 b kq - 3 5","prompt":"Black to move: attack the center","answer":"Nf6","hint":"Develop and attack e4"},
               {"fen":"r1bqk2r/pppp1ppp/2n2n2/2b5/2BpP3/2P2N2/PP3PPP/RNBQ1RK1 b kq - 0 6","prompt":"Black to move: castle and solidify","answer":"O-O","hint":"Get the king safe before complications arise"},
             ]},
        ]
    },
    "Classical": {
        "emoji":"♟️","color":"#3a7bd5",
        "description":"You follow sound principles — develop pieces, control the center, castle early.",
        "tip":"Study endgames — classical players convert small advantages in long games.",
        "openings":[
            {"name":"Ruy Lopez (Spanish Opening)","moves":"e4 e5 Nf3 Nc6 Bb5","pgn":"1.e4 e5 2.Nf3 Nc6 3.Bb5","difficulty":"Intermediate","players":"Karpov, Capablanca, Fischer","why":"The most classical opening — puts long-term pressure on the e5 pawn","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Ruy-Lopez-Opening",
             "puzzles":[
               {"fen":"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3","prompt":"Black to move: the most principled reply to Bb5","answer":"a6","hint":"Challenge the bishop immediately — the Morphy Defense"},
               {"fen":"r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4","prompt":"White to move: should the bishop retreat or capture?","answer":"Ba4","hint":"Maintain the pin on the knight — don't exchange yet"},
               {"fen":"r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5","prompt":"White to move: complete development","answer":"O-O","hint":"Castle and prepare to fight for the center"},
             ]},
            {"name":"Italian Game: Giuoco Piano","moves":"e4 e5 Nf3 Nc6 Bc4 Bc5 c3","pgn":"1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3","difficulty":"Beginner","players":"Carlsen, Greco","why":"Perfect for learning classical principles — simple, sound, and effective","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Italian-Game-Giuoco-Piano",
             "puzzles":[
               {"fen":"r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 4","prompt":"Black to move: the most natural development","answer":"Nf6","hint":"Develop the knight and attack e4"},
               {"fen":"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQ1RK1 b kq - 2 5","prompt":"Black to move: mirror White's plan","answer":"O-O","hint":"Castle and keep the position solid"},
               {"fen":"r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQ1RK1 b - - 0 6","prompt":"Black to move: fight for the center","answer":"d6","hint":"Solidify the e5 pawn and prepare Be6"},
             ]},
            {"name":"Four Knights Game","moves":"e4 e5 Nf3 Nc6 Nc3 Nf6","pgn":"1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6","difficulty":"Beginner","players":"Rubinstein, Marshall","why":"Symmetrical principled development — great for building opening fundamentals","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Four-Knights-Game",
             "puzzles":[
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4","prompt":"White to move: best principled 4th move","answer":"Bb5","hint":"The Spanish Four Knights — pin the knight"},
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4","prompt":"Black to move: mirror and challenge","answer":"Bb4","hint":"Pin White's knight back — the symmetrical variation"},
               {"fen":"r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5","prompt":"White to move: break the symmetry","answer":"O-O","hint":"Castle and prepare d4"},
             ]},
            {"name":"Petrov's Defense","moves":"e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4","pgn":"1.e4 e5 2.Nf3 Nf6 3.Nxe5 d6 4.Nf3 Nxe4","difficulty":"Intermediate","players":"Kramnik, Anand","why":"Ultra-solid reply — difficult to lose but requires precision to win","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Petrovs-Defense",
             "puzzles":[
               {"fen":"rnbqkb1r/pppp1ppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 2 3","prompt":"Black to move: the key Petrov move","answer":"d6","hint":"Attack the knight and recapture — don't take e4 immediately"},
               {"fen":"rnbqkb1r/ppp2ppp/3p4/4N3/4n3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 5","prompt":"White to move: win the knight back safely","answer":"d3","hint":"Attack the e4 knight and develop"},
               {"fen":"rnbqkb1r/ppp2ppp/3p4/8/4n3/3P4/PPP2PPP/RNBQKB1R b KQkq - 0 6","prompt":"Black to move: regroup","answer":"Nf6","hint":"The knight must retreat to a safe square"},
             ]},
            {"name":"Ruy Lopez: Berlin Defense","moves":"e4 e5 Nf3 Nc6 Bb5 Nf6","pgn":"1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6","difficulty":"Advanced","players":"Kramnik, Carlsen","why":"The 'Berlin Wall' — endgame-focused and extremely hard to crack","elo_min":1000,
             "chesscom":"https://www.chess.com/openings/Ruy-Lopez-Opening-Berlin-Defense",
             "puzzles":[
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4","prompt":"White to move: enter the Berlin endgame","answer":"O-O","hint":"Castle — the Berlin endgame is coming"},
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4","prompt":"Black to move: the Berlin endgame idea","answer":"Nxe4","hint":"Take the pawn — the endgame after Bxc6 dxc6 Nxe5 is approximately equal"},
               {"fen":"r1bqkb1r/pppp1ppp/2p2n2/4n3/4P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 6","prompt":"White to move: win material","answer":"Re1","hint":"Pin the knight on e5"},
             ]},
            {"name":"Scotch Game","moves":"e4 e5 Nf3 Nc6 d4 exd4 Nxd4","pgn":"1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Nxd4","difficulty":"Intermediate","players":"Kasparov, Carlsen","why":"Opens the center early — leads to rich tactical and positional play","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Scotch-Game",
             "puzzles":[
               {"fen":"r1bqkbnr/pppp1ppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4","prompt":"Black to move: fight for equality","answer":"Bc5","hint":"Develop the bishop and pressure the d4 knight"},
               {"fen":"r1bqk1nr/pppp1ppp/2n5/2b5/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5","prompt":"White to move: the critical choice","answer":"Be3","hint":"Defend the knight and prepare c3"},
               {"fen":"r1bqk1nr/pppp1ppp/2n5/2b5/3NP3/4B3/PPP2PPP/RN1QKB1R b KQkq - 2 5","prompt":"Black to move: maintain the tension","answer":"Qf6","hint":"Attack e5 and keep the bishop on c5"},
             ]},
            {"name":"Bishop's Opening","moves":"e4 e5 Bc4","pgn":"1.e4 e5 2.Bc4","difficulty":"Beginner","players":"Greco, Fischer","why":"Simple and principled — targets f7 immediately and keeps options flexible","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Bishops-Opening",
             "puzzles":[
               {"fen":"rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2","prompt":"Black to move: most natural reply","answer":"Nf6","hint":"Develop and attack e4"},
               {"fen":"rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR b KQkq - 3 3","prompt":"Black to move: mirror and challenge","answer":"Bc5","hint":"Develop the bishop symmetrically"},
               {"fen":"rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4","prompt":"Black to move: complete development","answer":"O-O","hint":"Castle and prepare for the middlegame"},
             ]},
            {"name":"Three Knights Game","moves":"e4 e5 Nf3 Nc6 Nc3","pgn":"1.e4 e5 2.Nf3 Nc6 3.Nc3","difficulty":"Beginner","players":"Spassky","why":"Rapid development with options to steer into sharp or solid lines","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Three-Knights-Opening",
             "puzzles":[
               {"fen":"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 3 3","prompt":"Black to move: best developing move","answer":"Nf6","hint":"Complete the symmetry — develop and attack e4"},
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4","prompt":"White to move: choose your setup","answer":"Bb5","hint":"Play the Spanish setup — pin the knight"},
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4","prompt":"Black to move: challenge the bishop","answer":"Nd4","hint":"Fork the bishop and f3 knight"},
             ]},
        ]
    },
    "Strategic": {
        "emoji":"🧠","color":"#2ecc71",
        "description":"You think long-term. Pawn structures and slow accumulation of advantages define your game.",
        "tip":"Study pawn structures — strategic players win by knowing which breaks to make.",
        "openings":[
            {"name":"Queen's Gambit Declined","moves":"d4 d5 c4 e6 Nc3 Nf6","pgn":"1.d4 d5 2.c4 e6 3.Nc3 Nf6","difficulty":"Intermediate","players":"Karpov, Kramnik, Carlsen","why":"Solid structure with rich positional play and long-term pressure","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Queens-Gambit-Declined",
             "puzzles":[
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4","prompt":"White to move: best plan in the QGD","answer":"Bg5","hint":"Pin the knight and create long-term pressure"},
               {"fen":"rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR w KQkq - 4 5","prompt":"White to move: fight for the center","answer":"e3","hint":"Solidify the center and prepare Bd3"},
               {"fen":"rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 4 7","prompt":"White to move: the Exchange variation","answer":"cxd5","hint":"Release the tension and play for a structural advantage"},
             ]},
            {"name":"King's Indian Defense","moves":"d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3","pgn":"1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3","difficulty":"Advanced","players":"Fischer, Kasparov","why":"Lets White take the center, then launches a fierce counterattack","elo_min":800,
             "chesscom":"https://www.chess.com/openings/Kings-Indian-Defense",
             "puzzles":[
               {"fen":"rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq - 0 6","prompt":"Black to move: the critical KID move","answer":"O-O","hint":"Castle and prepare the counterattack with e5"},
               {"fen":"rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 1 7","prompt":"White to move: set up the classical main line","answer":"Be2","hint":"Develop and prepare to castle"},
               {"fen":"rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 8","prompt":"White to move: fight for the center","answer":"O-O","hint":"Castle and then play d5 or dxe5"},
             ]},
            {"name":"English Opening","moves":"c4 e5 Nc3 Nf6 Nf3","pgn":"1.c4 e5 2.Nc3 Nf6 3.Nf3","difficulty":"Intermediate","players":"Karpov, Botvinnik","why":"Flexible hypermodern — delays the center fight for strategic maneuvering","elo_min":0,
             "chesscom":"https://www.chess.com/openings/English-Opening",
             "puzzles":[
               {"fen":"rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3","prompt":"White to move: develop flexibly","answer":"Nf3","hint":"Develop the knight and keep options open"},
               {"fen":"rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N2N2/PP1PPPPP/R1BQKB1R b KQkq - 3 3","prompt":"Black to move: challenge or develop?","answer":"Nc6","hint":"Develop and support the e5 pawn"},
               {"fen":"r1bqkb1r/pppp1ppp/2n2n2/4p3/2P5/2N2N2/PP1PPPPP/R1BQKB1R w KQkq - 4 4","prompt":"White to move: fight for d5","answer":"d4","hint":"Strike in the center"},
             ]},
            {"name":"Nimzo-Indian Defense","moves":"d4 Nf6 c4 e6 Nc3 Bb4","pgn":"1.d4 Nf6 2.c4 e6 3.Nc3 Bb4","difficulty":"Intermediate","players":"Nimzowitsch, Kasparov, Karpov","why":"Controls the center with pieces rather than pawns — rich strategic battles","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Nimzo-Indian-Defense",
             "puzzles":[
               {"fen":"rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4","prompt":"White to move: choose your Nimzo system","answer":"Qc2","hint":"The classical system — prevent doubling and keep the bishop pair"},
               {"fen":"rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PPQ1PPPP/R1B1KBNR b KQkq - 3 4","prompt":"Black to move: take the pawn or retreat?","answer":"O-O","hint":"Castle first — the bishop will come back"},
               {"fen":"rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2N1P3/PPQ2PPP/R1B1KBNR w KQ - 2 5","prompt":"White to move: fight for d4","answer":"Bd2","hint":"Develop and protect d4 against Bxc3"},
             ]},
            {"name":"Catalan Opening","moves":"d4 Nf6 c4 e6 g3 d5 Bg2","pgn":"1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2","difficulty":"Advanced","players":"Kramnik, Carlsen","why":"Long-term pressure down the c-file and d-diagonal — positional masterclass","elo_min":1000,
             "chesscom":"https://www.chess.com/openings/Catalan-Opening",
             "puzzles":[
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR w KQkq - 0 4","prompt":"White to move: fianchetto the bishop","answer":"Bg2","hint":"Complete the Catalan setup — the bishop on g2 is the key piece"},
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PPBP/RNBQK1NR b KQkq - 1 4","prompt":"Black to move: take the gambit pawn?","answer":"dxc4","hint":"The Open Catalan — take the pawn and try to hold it"},
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/8/2pP4/6P1/PP2PPBP/RNBQK1NR w KQkq - 0 5","prompt":"White to move: regain the pawn or build pressure?","answer":"Nf3","hint":"Develop first — the c4 pawn will be recovered with Qa4 or Ne5"},
             ]},
            {"name":"Reti Opening","moves":"Nf3 d5 c4","pgn":"1.Nf3 d5 2.c4","difficulty":"Intermediate","players":"Reti, Karpov","why":"Hypermodern — attacks the center from the flanks with pieces, not pawns","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Reti-Opening",
             "puzzles":[
               {"fen":"rnbqkbnr/ppp1pppp/8/3p4/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 0 2","prompt":"Black to move: defend or advance?","answer":"d4","hint":"The aggressive pawn advance — gain space in the center"},
               {"fen":"rnbqkbnr/ppp1pppp/8/8/2Pp4/5N2/PP1PPPPP/RNBQKB1R w KQkq - 0 3","prompt":"White to move: best response to d4","answer":"e3","hint":"Challenge the pawn and open lines for your pieces"},
               {"fen":"rnbqkbnr/ppp1pppp/8/8/2P5/4pN2/PP1P1PPP/RNBQKB1R w KQkq - 0 4","prompt":"White to move: what if Black promotes the pawn?","answer":"Nxe3","hint":"Recapture and maintain material balance"},
             ]},
            {"name":"London System","moves":"d4 d5 Nf3 Nf6 Bf4 e6 e3","pgn":"1.d4 d5 2.Nf3 Nf6 3.Bf4 e6 4.e3","difficulty":"Beginner","players":"Carlsen, Giri","why":"Simple, solid setup that avoids heavy theory — grind opponents down slowly","elo_min":0,
             "chesscom":"https://www.chess.com/openings/London-System",
             "puzzles":[
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R w KQkq - 2 4","prompt":"White to move: solidify the London setup","answer":"e3","hint":"Protect the bishop and prepare Bd3"},
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R b KQkq - 0 4","prompt":"Black to move: challenge the London","answer":"c5","hint":"Strike at the d4 base immediately"},
               {"fen":"rnbqkb1r/pp3ppp/4pn2/2pp4/3P1B2/4PN2/PPP2PPP/RN1QKB1R w KQkq - 0 5","prompt":"White to move: maintain control","answer":"c3","hint":"Solidify d4 and prepare to develop the queen's knight"},
             ]},
            {"name":"Queen's Gambit Accepted","moves":"d4 d5 c4 dxc4 Nf3 Nf6","pgn":"1.d4 d5 2.c4 dxc4 3.Nf3 Nf6","difficulty":"Intermediate","players":"Anand, Spassky","why":"Taking the gambit pawn leads to active piece play and open positions","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Queens-Gambit-Accepted",
             "puzzles":[
               {"fen":"rnbqkb1r/ppp1pppp/5n2/8/2pP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4","prompt":"White to move: win back the pawn","answer":"e3","hint":"Prepare Bxc4 — develop the bishop and fight for the c4 pawn"},
               {"fen":"rnbqkb1r/ppp1pppp/5n2/8/2pP4/4PN2/PP3PPP/RNBQKB1R b KQkq - 0 4","prompt":"Black to move: hold the extra pawn?","answer":"e6","hint":"Consolidate and prepare to develop — it's hard to hold c4 long term"},
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/8/2pP4/4PN2/PP3PPP/RNBQKB1R w KQkq - 0 5","prompt":"White to move: recapture the pawn","answer":"Bxc4","hint":"Recover the pawn and develop with tempo"},
             ]},
        ]
    },
    "Solid": {
        "emoji":"🛡️","color":"#9b59b6",
        "description":"You value safety and reliability. You avoid complications and build from a secure foundation.",
        "tip":"Work on converting small advantages — solid players win by slowly outplaying opponents.",
        "openings":[
            {"name":"Caro-Kann Defense","moves":"e4 c6 d4 d5 Nc3 dxe4 Nxe4","pgn":"1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4","difficulty":"Intermediate","players":"Petrosian, Karpov, Anand","why":"Rock-solid structure — avoids early complications without being passive","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Caro-Kann-Defense",
             "puzzles":[
               {"fen":"rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3","prompt":"White to move: most aggressive Caro-Kann response","answer":"Nc3","hint":"Develop and support the advance of e5"},
               {"fen":"rnbqkbnr/pp2pppp/2p5/8/3Pn3/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 4","prompt":"White to move: deal with the knight","answer":"Nf3","hint":"Develop and attack the e4 knight"},
               {"fen":"rnbqkb1r/pp2pppp/2p2n2/8/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 2 5","prompt":"Black to move: where does the bishop go?","answer":"Bf5","hint":"Develop actively outside the pawn chain — the trademark Caro-Kann move"},
             ]},
            {"name":"French Defense","moves":"e4 e6 d4 d5 Nc3 Nf6","pgn":"1.e4 e6 2.d4 d5 3.Nc3 Nf6","difficulty":"Intermediate","players":"Nimzowitsch, Uhlmann","why":"Creates a solid pawn chain and leads to rich strategic battles","elo_min":0,
             "chesscom":"https://www.chess.com/openings/French-Defense",
             "puzzles":[
               {"fen":"rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3","prompt":"Black to move: fight for the center","answer":"Nf6","hint":"Develop and attack e4"},
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4","prompt":"White to move: the critical decision","answer":"e5","hint":"Advance — the Advance Variation creates a space advantage"},
               {"fen":"rnbqkb1r/ppp2ppp/4p3/3pPn2/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 5","prompt":"White to move: challenge the strong f5 knight","answer":"Nce2","hint":"Support d4 and prepare to kick the knight with g4"},
             ]},
            {"name":"Slav Defense","moves":"d4 d5 c4 c6 Nf3 Nf6 Nc3","pgn":"1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3","difficulty":"Intermediate","players":"Kramnik, Anand","why":"Extremely reliable against the Queen's Gambit — solid with counterplay","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Slav-Defense",
             "puzzles":[
               {"fen":"rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 1 4","prompt":"Black to move: develop the light-squared bishop","answer":"Bf5","hint":"The classical Slav — the bishop escapes before e6 is played"},
               {"fen":"rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 1 4","prompt":"White to move: challenge the Slav structure","answer":"cxd5","hint":"Exchange pawns and fight for the c5 square"},
               {"fen":"rnbqkb1r/pp2pppp/2p2n2/5B2/2PP4/2N2N2/PP2PPPP/R2QKB1R b KQkq - 1 5","prompt":"Black to move: deal with Bf5","answer":"cxd5","hint":"Recapture and free the position"},
             ]},
            {"name":"Scandinavian Defense","moves":"e4 d5 exd5 Qxd5 Nc3 Qa5","pgn":"1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5","difficulty":"Beginner","players":"Tiviakov, Carlsen (occasional)","why":"Immediate central challenge — simple and easy to learn for all levels","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Scandinavian-Defense",
             "puzzles":[
               {"fen":"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2","prompt":"White to move: take the pawn","answer":"exd5","hint":"Accept the challenge in the center"},
               {"fen":"rnbqkbnr/ppp1pppp/8/3Q4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3","prompt":"Black to move: bring the queen or recapture with the knight?","answer":"Qxd5","hint":"Take with the queen — the Scandinavian plan"},
               {"fen":"rnb1kbnr/ppp1pppp/8/q7/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4","prompt":"White to move: chase the queen","answer":"d4","hint":"Build the pawn center and gain tempo on the queen"},
             ]},
            {"name":"Semi-Slav Defense","moves":"d4 d5 c4 c6 Nf3 Nf6 Nc3 e6","pgn":"1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6","difficulty":"Advanced","players":"Anand, Topalov","why":"One of the most theoretically rich defenses — solid yet full of counterplay","elo_min":800,
             "chesscom":"https://www.chess.com/openings/Semi-Slav-Defense",
             "puzzles":[
               {"fen":"rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5","prompt":"White to move: enter the Meran or Anti-Meran","answer":"e3","hint":"The Meran — set up Be2 and castle"},
               {"fen":"rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R b KQkq - 0 5","prompt":"Black to move: the Meran counter","answer":"Nbd7","hint":"Develop the knight and prepare b5"},
               {"fen":"rnbqkb1r/p4ppp/2p1pn2/1p1p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq b6 0 6","prompt":"White to move: fight for the center","answer":"cxd5","hint":"Exchange and then cxb5 — the main Meran line"},
             ]},
            {"name":"King's Indian Attack","moves":"Nf3 d5 g3 Nf6 Bg2 e6 O-O","pgn":"1.Nf3 d5 2.g3 Nf6 3.Bg2 e6 4.O-O","difficulty":"Intermediate","players":"Fischer, Karpov","why":"White sets up a safe king and waits for the right moment to strike","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Kings-Indian-Attack",
             "puzzles":[
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 2 4","prompt":"White to move: complete the KIA setup","answer":"O-O","hint":"Castle and prepare d3 and Nbd2"},
               {"fen":"rnbqkb1r/ppp2ppp/4pn2/3p4/8/5NP1/PPPPPPBP/RNBQ1RK1 b kq - 3 4","prompt":"Black to move: natural development","answer":"Be7","hint":"Develop and prepare to castle"},
               {"fen":"rnbq1rk1/ppp1bppp/4pn2/3p4/8/3P1NP1/PPP1PPBP/RNBQ1RK1 w - - 4 6","prompt":"White to move: start the kingside plan","answer":"e4","hint":"Strike in the center — the KIA's main idea"},
             ]},
            {"name":"Dutch Defense: Stonewall","moves":"d4 f5 c4 Nf6 g3 e6 Bg2 d5","pgn":"1.d4 f5 2.c4 Nf6 3.g3 e6 4.Bg2 d5","difficulty":"Intermediate","players":"Botvinnik, Short","why":"A fortress structure — difficult to break down, ideal for patient players","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Dutch-Defense-Stonewall-Variation",
             "puzzles":[
               {"fen":"rnbqkb1r/ppppp1pp/5n2/5p2/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3","prompt":"White to move: best against the Dutch","answer":"g3","hint":"Fianchetto the bishop — the strongest plan against the Dutch"},
               {"fen":"rnbqkb1r/ppppp1pp/5n2/5p2/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3","prompt":"Black to move: set up the Stonewall","answer":"e6","hint":"Play e6 before d5 to set up the Stonewall pawn structure"},
               {"fen":"rnbqkb1r/ppp3pp/4pn2/3p1p2/2PP4/6P1/PP2PPBP/RNBQK1NR w KQkq - 0 5","prompt":"White to move: attack the Stonewall","answer":"Nf3","hint":"Develop and fight for the e5 square"},
             ]},
            {"name":"Caro-Kann: Classical","moves":"e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5","pgn":"1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5","difficulty":"Intermediate","players":"Karpov, Leko","why":"Keeps the light-squared bishop active — a refined, solid approach","elo_min":0,
             "chesscom":"https://www.chess.com/openings/Caro-Kann-Defense-Classical-Variation",
             "puzzles":[
               {"fen":"rnbqkbnr/pp2pppp/2p5/5b2/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 1 5","prompt":"White to move: most aggressive response to Bf5","answer":"Ng3","hint":"Attack the bishop and force it to commit"},
               {"fen":"rnbqkbnr/pp2pppp/2p5/6b1/3P4/6N1/PPP2PPP/R1BQKBNR b KQkq - 2 5","prompt":"Black to move: best bishop retreat","answer":"Bg6","hint":"Retreat safely and keep the bishop active"},
               {"fen":"rnbqkbnr/pp2pppp/2p3b1/8/3P4/6N1/PPP1PPPP/R1BQKBNR w KQkq - 3 6","prompt":"White to move: attack the g6 bishop","answer":"h4","hint":"Challenge the bishop with a flank pawn advance"},
             ]},
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

def get_eco_family(eco):
    """ECO code (A00-E99) → opening family. Most reliable method."""
    if not eco or len(eco)<2: return None
    try: letter=eco[0].upper(); num=int(eco[1:])
    except: return None
    if letter=='A':
        if num<=9: return "Aggressive"
        if num<=39: return "Strategic"   # English
        if num<=44: return "Aggressive"  # Benoni
        if num<=49: return "Solid"       # Dutch
        return "Aggressive"
    if letter=='B':
        if num<=9: return "Aggressive"   # Scandinavian, Alekhine
        if num<=19: return "Solid"       # Caro-Kann
        return "Aggressive"              # Sicilian (B20-B99)
    if letter=='C':
        if num<=19: return "Classical"   # Spanish/Italian early
        if num<=29: return "Aggressive"  # King's Gambit, Vienna
        if num<=49: return "Classical"   # Italian, Four Knights
        if num<=79: return "Classical"   # Ruy Lopez
        return "Solid"                   # French
    if letter=='D':
        if num<=49: return "Strategic"   # QGD, Slav
        return "Strategic"               # Grunfeld
    if letter=='E':
        if num<=19: return "Strategic"   # Catalan, Bogo, QID
        if num<=59: return "Strategic"   # Nimzo-Indian
        return "Aggressive"              # King's Indian, Benoni
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
        tc=tag("TimeClass") or "blitz"; opening=tag("Opening"); eco=tag("ECO"); term=tag("Termination") or ""
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
        family=get_family_from_moves(moves_text) or get_eco_family(eco) or get_opening_family(opening)
        castled=1 if "O-O" in moves_text else 0
        records.append({"username":username,"player_elo":player_elo,"opponent_elo":opp_elo,
            "elo_diff":player_elo-opp_elo,"played_as_enc":1 if played_as=="white" else 0,
            "time_class":tc,"time_class_enc":{"bullet":0,"blitz":1,"rapid":2,"daily":3}.get(tc,1),
            "opening_name":opening or "","opening_family":family,"outcome":outcome,
            "num_moves":max(len(moves),1),"decisive":1 if outcome!="draw" else 0,
            "resigned":1 if "resign" in term.lower() else 0,
            "castled":castled,
            "is_bullet":1 if tc=="bullet" else 0})
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
          "num_moves","decisive","resigned","win_rate","draw_rate","decisive_rate",
          "resign_rate","aggression_score","castle_rate","bullet_ratio","opening_diversity"]

def train_models(records, progress_cb=None):
    df=pd.DataFrame(records); df=df[df["opening_family"].notna()&(df["opening_family"]!="")].copy()
    if len(df)<30: return None,f"Only {len(df)} labeled games. Need at least 30."
    # ── Per-player aggregate features (Lab 2 style) ──
    agg=df.groupby("username").agg(
        win_rate=("outcome",lambda x:(x=="win").mean()),
        draw_rate=("outcome",lambda x:(x=="draw").mean()),
        decisive_rate=("decisive","mean"),
        resign_rate=("resigned","mean"),
        aggression_score=("num_moves",lambda x:1/(x.mean()+1)),
        castle_rate=("castled","mean"),           # Lab feature: how often does the player castle?
        bullet_ratio=("is_bullet","mean"),        # Lab feature: fraction of bullet games
        opening_diversity=("opening_name",lambda x: min(x.nunique()/max(len(x),1), 1.0)),  # Lab feature: variety of openings
    ).reset_index()
    df=df.merge(agg,on="username")
    feats=[f for f in FEATURES if f in df.columns]
    X=df[feats].fillna(0); y=df["opening_family"]
    if progress_cb: progress_cb(f"📊 Family distribution: {y.value_counts().to_dict()}")
    if y.nunique()<2: return None,"Not enough opening variety — most games share the same opening family. Try adding more usernames."
    min_class_count = y.value_counts().min()
    use_stratify = y if min_class_count >= 5 else None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=use_stratify)
    scaler=StandardScaler(); Xtr_s=scaler.fit_transform(X_train); Xte_s=scaler.transform(X_test)

    # ── Lab 9: K-Means unsupervised clustering (used as an extra validation signal) ──
    if progress_cb: progress_cb("🔍 Running K-Means clustering (Lab 9)...")
    kmeans=KMeans(n_clusters=4,random_state=42,n_init=10)
    cluster_labels=kmeans.fit_predict(scaler.transform(X))
    cluster_map={i:y.iloc[np.where(cluster_labels==i)[0]].mode()[0]
                 for i in range(4) if len(np.where(cluster_labels==i)[0])>0}
    if progress_cb: progress_cb(f"  ✓ K-Means cluster → style mapping: {cluster_map}")

    # ── Lab 3/4/5/7: Supervised models (optimised for speed on Railway) ──
    from sklearn.linear_model import LogisticRegression
    mdls={
        "Naive Bayes":     (GaussianNB(), True),                                                        # Lab 3
        "Decision Tree":   (DecisionTreeClassifier(max_depth=6,random_state=42), False),                # Lab 4
        "Random Forest":   (RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1), False),  # Lab 5
        "Logistic Reg":    (LogisticRegression(max_iter=300,random_state=42,n_jobs=-1), True),          # Lab 6
        "SVM Linear":      (SVC(kernel="linear",C=1.0,probability=True,random_state=42,max_iter=2000),True), # Lab 7
    }
    results={}; best_acc,best_name,best_model=0,None,None
    for name,(model,scaled) in mdls.items():
        t0=time.time()
        if progress_cb: progress_cb(f"🤖 Training {name}...")
        Xtr=Xtr_s if scaled else X_train; Xte=Xte_s if scaled else X_test
        model.fit(Xtr,y_train); acc=accuracy_score(y_test,model.predict(Xte))
        results[name]=round(acc*100,1)
        if progress_cb: progress_cb(f"  ✓ {name}: {round(acc*100,1)}% ({round(time.time()-t0,1)}s)")
        if acc>best_acc: best_acc,best_name,best_model=acc,name,model
    rf=mdls["Random Forest"][0]
    importances={f:round(float(v),4) for f,v in zip(feats,rf.feature_importances_)}
    saved={"model":best_model,"model_name":best_name,"scaler":scaler,"features":feats,
           "classes":sorted(y.unique()),"results":results,"importances":importances,
           "total_games":len(df),"family_counts":df["opening_family"].value_counts().to_dict(),
           "kmeans":kmeans,"cluster_map":cluster_map}
    with open("best_model.pkl","wb") as f: pickle.dump(saved,f)
    if progress_cb: progress_cb(f"✅ Best: {best_name} ({round(best_acc*100,1)}%)")
    return saved,None

def validate_username(username):
    """Check if a Chess.com username exists. Returns (True, None) or (False, error_msg)."""
    try:
        r=requests.get(f"https://api.chess.com/pub/player/{username}",headers=HEADERS,timeout=8)
        if r.status_code==404: return False,f"Player '{username}' not found on Chess.com. Check the spelling."
        if r.status_code!=200: return False,f"Chess.com returned an error ({r.status_code}). Try again later."
        return True,None
    except requests.Timeout: return False,"Request timed out. Chess.com may be slow — try again."
    except Exception as e: return False,f"Could not reach Chess.com: {e}"

def pick_openings(family, player_stats, n=3):
    """Pick n openings from the family, varied by player ELO and stats."""
    import random
    pool=OPENING_DB[family]["openings"]
    elo=player_stats.get("avg_elo",1000)
    # Filter by ELO appropriateness
    eligible=[o for o in pool if o.get("elo_min",0)<=elo]
    if len(eligible)<n: eligible=pool  # fall back to all if too few
    # Seed random with username+elo so same player always gets same recs but differs from others
    rng=random.Random(hash((player_stats.get("username",""),round(elo,-2))))
    # Weight towards harder openings for higher ELO players
    if elo>1500:
        advanced=[o for o in eligible if o["difficulty"]=="Advanced"]
        beginner=[o for o in eligible if o["difficulty"]=="Beginner"]
        mid=[o for o in eligible if o["difficulty"]=="Intermediate"]
        weighted=advanced*3+mid*2+beginner*1
    else:
        advanced=[o for o in eligible if o["difficulty"]=="Advanced"]
        beginner=[o for o in eligible if o["difficulty"]=="Beginner"]
        mid=[o for o in eligible if o["difficulty"]=="Intermediate"]
        weighted=beginner*3+mid*2+advanced*1
    if len(weighted)<n: weighted=eligible
    seen=set(); picks=[]
    rng.shuffle(weighted)
    for o in weighted:
        if o["name"] not in seen:
            seen.add(o["name"]); picks.append(o)
        if len(picks)==n: break
    return picks

def recommend_for_user(username):
    if not os.path.exists("best_model.pkl"): return None,"Model not trained yet."
    valid,verr=validate_username(username)
    if not valid: return None,verr
    with open("best_model.pkl","rb") as f: saved=pickle.load(f)
    records=fetch_all_games([username],months=2)
    if not records: return None,f"No games found for '{username}'. They may have no recent games."
    df=pd.DataFrame(records)
    agg={"player_elo":df["player_elo"].mean(),"opponent_elo":df["opponent_elo"].mean(),
         "elo_diff":df["elo_diff"].mean(),"played_as_enc":df["played_as_enc"].mean(),
         "time_class_enc":df["time_class_enc"].median(),"num_moves":df["num_moves"].mean(),
         "decisive":df["decisive"].mean(),"resigned":df["resigned"].mean(),
         "win_rate":(df["outcome"]=="win").mean(),"draw_rate":(df["outcome"]=="draw").mean(),
         "decisive_rate":df["decisive"].mean(),"resign_rate":df["resigned"].mean(),
         "aggression_score":1/(df["num_moves"].mean()+1),
         "castle_rate":df["castled"].mean() if "castled" in df.columns else 0.5,
         "bullet_ratio":df["is_bullet"].mean() if "is_bullet" in df.columns else 0.0,
         "opening_diversity":min(df["opening_name"].nunique()/max(len(df),1),1.0) if "opening_name" in df.columns else 0.5}
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
    player_stats={"username":username,"avg_elo":round(agg["player_elo"]),"win_rate":round(agg["win_rate"]*100,1),
                  "draw_rate":round(agg["draw_rate"]*100,1),"avg_moves":round(agg["num_moves"],1),
                  "resign_rate":round(agg["resign_rate"]*100,1),"games_analyzed":len(df)}
    chosen_openings=pick_openings(pred, player_stats)
    return {"username":username,"predicted_family":pred,"probabilities":probs,
            "openings":chosen_openings,
            "style_info":{"emoji":info["emoji"],"color":info["color"],"description":info["description"],"tip":info["tip"]},
            "player_stats":player_stats,
            "model_used":model_name,"model_accuracy":saved["results"].get(model_name,0)},None

training_state={"status":"idle","logs":[],"result":None,"error":None}

def run_training():
    global training_state
    training_state={"status":"training","logs":[],"result":None,"error":None}
    # ── Carefully chosen players — each is a clear example of their style ──
    AGGRESSIVE_USERS=[
        "hikaru",            # Super GM, ultra-sharp blitz/bullet tactician
        "danielnaroditsky",  # Speed chess specialist, loves sharp lines
        "nihalsarin",        # Young attacking Indian GM
        "firouzja2003",      # World #2, extremely aggressive style
        "rpragchess",        # Praggnanandhaa — sharp tactical prodigy
        "lachesisq",         # Danya's alt account — pure blitz aggression
    ]
    CLASSICAL_USERS=[
        "gothamchess",       # Levy Rozman — textbook principled play
        "GMBenjaminFinegold",# Classical positional GM educator
        "yasser-seirawan",   # Legend, pure classical style
        "JohnBartholomewChess", # Classical educator, instructional games
    ]
    STRATEGIC_USERS=[
        "magnuscarlsen",     # World Champion, strategic mastery
        "fabianocaruana",    # Elite GM, deep preparation and strategy
        "gmwso",             # Wesley So — positional genius
        "liem_le",           # Le Quang Liem — strategic top GM
        "gmkovalev",         # Strategic European GM
        "alireza2003",       # Firouzja's strategic side — varied
    ]
    SOLID_USERS=[
        "anishgiri",         # Draw master, ultra-solid Caro-Kann player
        "levonAronian",      # Solid foundations with occasional creativity
        "penguingm",         # FM streamer, reliable solid repertoire
        "chessbrahs",        # Aman Hambleton — solid positional streamer
        "dereque",           # Solid club-level educator
        "GMHannes",          # Solid Scandinavian defensive style
    ]
    TRAIN_USERS=AGGRESSIVE_USERS+CLASSICAL_USERS+STRATEGIC_USERS+SOLID_USERS
    # Build a lookup so each player's known style overrides move-based detection during training
    KNOWN_STYLES={u:"Aggressive" for u in AGGRESSIVE_USERS}
    KNOWN_STYLES.update({u:"Classical" for u in CLASSICAL_USERS})
    KNOWN_STYLES.update({u:"Strategic" for u in STRATEGIC_USERS})
    KNOWN_STYLES.update({u:"Solid" for u in SOLID_USERS})
    def log(msg): training_state["logs"].append(msg)
    log("🔄 Connecting to Chess.com API...")
    records=fetch_all_games(TRAIN_USERS,months=3,progress_cb=log)
    # Override opening_family with known style for training players — much cleaner labels
    for r in records:
        if r.get("username","").lower() in {k.lower():v for k,v in KNOWN_STYLES.items()}:
            r["opening_family"]=KNOWN_STYLES.get(r["username"], r["opening_family"])
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
.octa{margin-top:14px;}
.btn-chesscom{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;background:rgba(129,211,73,0.12);color:#81d349;border:1px solid rgba(129,211,73,0.25);border-radius:8px;text-decoration:none;font-size:12px;font-weight:600;font-family:'DM Sans',sans-serif;transition:all 0.2s;}
.btn-chesscom:hover{background:rgba(129,211,73,0.22);border-color:rgba(129,211,73,0.5);}
.otabs{display:flex;gap:6px;padding:12px 12px 0;border-bottom:1px solid var(--border);background:#181412;width:100%;}
.otab{background:none;border:none;color:var(--muted);font-size:12px;font-family:'DM Sans',sans-serif;padding:6px 12px;cursor:pointer;border-radius:6px 6px 0 0;transition:all 0.15s;}
.otab.active{background:rgba(255,255,255,0.08);color:var(--text);}
.pcard{background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:10px;}
.pnum{font-family:'DM Mono',monospace;font-size:10px;color:var(--gold);margin-bottom:6px;}
.pprompt{font-size:13px;color:var(--text);line-height:1.5;margin-bottom:10px;}
.phint{font-size:12px;color:#c9a84c;background:rgba(201,168,76,0.08);padding:8px 10px;border-radius:6px;margin-bottom:8px;}
.pans{font-size:13px;color:#2ecc71;background:rgba(46,204,113,0.08);padding:8px 10px;border-radius:6px;margin-bottom:8px;}
.pbtnrow{display:flex;gap:8px;}
/* Board area */
.bwrap{background:#181412;display:flex;flex-direction:column;align-items:center;padding:0 16px 16px;gap:10px;border-left:1px solid var(--border);overflow:auto;}
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
.puzzle-panel{width:100%;padding:12px 0;overflow-y:auto;}
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
  <div id="tprog" style="display:none;margin-top:16px;">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div class="logbox" id="logbox"></div>
      <div style="background:#0a0a0c;border:1px solid var(--border);border-radius:10px;padding:12px;">
        <div style="font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);margin-bottom:8px;">LIVE ACCURACY CHART</div>
        <canvas id="trainChart" height="120"></canvas>
      </div>
    </div>
  </div>
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

// ── CHESS SOUND ENGINE ──────────────────────────────────────
const AudioCtx=window.AudioContext||window.webkitAudioContext;
let actx=null;
function getACtx(){if(!actx)actx=new AudioCtx();return actx;}

function playMoveSound(type='move'){
  try{
    const ac=getACtx();
    if(type==='move'){
      // Wooden piece-on-board thud — two-layer: click + resonance
      const buf=ac.createBuffer(1,ac.sampleRate*0.18,ac.sampleRate);
      const d=buf.getChannelData(0);
      for(let i=0;i<d.length;i++){
        const t=i/ac.sampleRate;
        d[i]=(Math.random()*2-1)*Math.exp(-t*55)*(0.6+0.4*Math.exp(-t*120));
      }
      const src=ac.createBufferSource();src.buffer=buf;
      const g=ac.createGain();g.gain.setValueAtTime(0.55,ac.currentTime);
      // Slight low-pass to make it warmer/woodier
      const f=ac.createBiquadFilter();f.type='lowpass';f.frequency.value=2800;
      src.connect(f);f.connect(g);g.connect(ac.destination);
      src.start();
    } else if(type==='capture'){
      // Heavier thud for captures
      const buf=ac.createBuffer(1,ac.sampleRate*0.22,ac.sampleRate);
      const d=buf.getChannelData(0);
      for(let i=0;i<d.length;i++){
        const t=i/ac.sampleRate;
        d[i]=(Math.random()*2-1)*Math.exp(-t*38)*(0.9+0.4*Math.exp(-t*90));
      }
      const src=ac.createBufferSource();src.buffer=buf;
      const g=ac.createGain();g.gain.setValueAtTime(0.75,ac.currentTime);
      const f=ac.createBiquadFilter();f.type='lowpass';f.frequency.value=2200;
      src.connect(f);f.connect(g);g.connect(ac.destination);
      src.start();
    } else if(type==='check'){
      // Higher pitched alert tone
      const osc=ac.createOscillator();const g=ac.createGain();
      osc.connect(g);g.connect(ac.destination);
      osc.frequency.setValueAtTime(880,ac.currentTime);
      osc.frequency.exponentialRampToValueAtTime(660,ac.currentTime+0.12);
      g.gain.setValueAtTime(0.3,ac.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001,ac.currentTime+0.22);
      osc.start();osc.stop(ac.currentTime+0.22);
    } else if(type==='start'){
      // Soft thud for board reset
      const buf=ac.createBuffer(1,ac.sampleRate*0.12,ac.sampleRate);
      const d=buf.getChannelData(0);
      for(let i=0;i<d.length;i++){
        const t=i/ac.sampleRate;
        d[i]=(Math.random()*2-1)*Math.exp(-t*80)*0.4;
      }
      const src=ac.createBufferSource();src.buffer=buf;
      const g=ac.createGain();g.gain.value=0.35;
      const f=ac.createBiquadFilter();f.type='lowpass';f.frequency.value=3200;
      src.connect(f);f.connect(g);g.connect(ac.destination);
      src.start();
    }
  }catch(e){}
}

function detectMoveType(boardBefore,boardAfter){
  if(!boardBefore||!boardAfter)return'move';
  // Count pieces on boardAfter vs boardBefore — if fewer pieces, it's a capture
  let before=0,after=0;
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    if(boardBefore[r][c])before++;
    if(boardAfter[r][c])after++;
  }
  return after<before?'capture':'move';
}
// ── END SOUND ENGINE ─────────────────────────────────────────

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
  const prev=s.currentMove;
  s.currentMove=Math.max(0,Math.min(s.boards.length-1,s.currentMove+dir));
  if(s.currentMove!==prev&&dir!==0){
    const type=detectMoveType(s.boards[prev],s.boards[s.currentMove]);
    playMoveSound(type);
  }
  renderBoard(id,s.boards[s.currentMove],null,null);
  updateCtr(id);
}

function resetBoard(id){const s=BS[id];if(!s)return;s.currentMove=0;renderBoard(id,s.boards[0],null,null);updateCtr(id);playMoveSound('start');}

function playAll(id){
  const s=BS[id];if(!s)return;
  s.currentMove=0;renderBoard(id,s.boards[0],null,null);updateCtr(id);playMoveSound('start');
  const iv=setInterval(()=>{
    if(s.currentMove>=s.boards.length-1){clearInterval(iv);return;}
    const prev=s.currentMove;
    s.currentMove++;
    const type=detectMoveType(s.boards[prev],s.boards[s.currentMove]);
    playMoveSound(type);
    renderBoard(id,s.boards[s.currentMove],null,null);updateCtr(id);
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

// Live training chart data
let chartData={labels:[],values:[]};
let chartCtx=null;

function initChart(){
  const canvas=document.getElementById("trainChart");
  if(!canvas)return;
  chartCtx=canvas.getContext("2d");
  chartData={labels:[],values:[]};
  drawChart();
}

function drawChart(){
  if(!chartCtx)return;
  const canvas=document.getElementById("trainChart");
  const W=canvas.width=canvas.offsetWidth; const H=canvas.height=120;
  chartCtx.clearRect(0,0,W,H);
  // Background
  chartCtx.fillStyle="#0a0a0c"; chartCtx.fillRect(0,0,W,H);
  // Grid lines
  [25,50,75,100].forEach(v=>{
    const y=H-(v/100)*H;
    chartCtx.strokeStyle="rgba(255,255,255,0.05)"; chartCtx.lineWidth=1;
    chartCtx.beginPath(); chartCtx.moveTo(0,y); chartCtx.lineTo(W,y); chartCtx.stroke();
    chartCtx.fillStyle="rgba(255,255,255,0.2)"; chartCtx.font="9px DM Mono,monospace";
    chartCtx.fillText(v+"%",2,y-2);
  });
  if(chartData.values.length<1)return;
  const COLORS={"Naive Bayes":"#d4537e","Decision Tree":"#378add","Random Forest":"#2ecc71","Logistic Reg":"#ef9f27","SVM Linear":"#7f77dd"};
  const n=chartData.labels.length;
  const barW=Math.min(40, (W-20)/Math.max(n,1));
  chartData.labels.forEach((label,i)=>{
    const acc=chartData.values[i];
    const x=20+i*(barW+6);
    const barH=(acc/100)*(H-20);
    const col=COLORS[label]||"#c9a84c";
    // Bar
    chartCtx.fillStyle=col+"33"; chartCtx.fillRect(x,H-barH,barW,barH);
    chartCtx.fillStyle=col; chartCtx.fillRect(x,H-barH,barW,4);
    // Label
    chartCtx.fillStyle=col; chartCtx.font="bold 11px DM Mono,monospace";
    chartCtx.fillText(acc+"%",x,H-barH-4);
    // Model name — abbreviated
    chartCtx.fillStyle="rgba(255,255,255,0.4)"; chartCtx.font="9px DM Mono,monospace";
    const short=label.replace("Random Forest","RF").replace("Decision Tree","DT").replace("Naive Bayes","NB").replace("Logistic Reg","LR").replace("SVM Linear","SVM");
    chartCtx.fillText(short,x,H-2);
  });
}

function pollTraining(){
  const lb=document.getElementById("logbox");
  initChart();
  const iv=setInterval(async()=>{
    const st=await fetch("/api/train/status").then(x=>x.json());
    lb.innerHTML=st.logs.map(l=>`<p class="${l.startsWith("✅")?"ok":""}">${l}</p>`).join("");
    lb.scrollTop=lb.scrollHeight;
    // Parse accuracy results from logs for live chart
    st.logs.forEach(line=>{
      const m=line.match(/✓ (.+): ([\d.]+)%/);
      if(m){
        const name=m[1].trim(); const acc=parseFloat(m[2]);
        const idx=chartData.labels.indexOf(name);
        if(idx===-1){chartData.labels.push(name);chartData.values.push(acc);}
        else chartData.values[idx]=acc;
        drawChart();
      }
    });
    if(st.status==="done"){
      clearInterval(iv);
      showMR(st.result.model_results,st.result.best_model,st.result.total_games);
      document.getElementById("train-btn").innerHTML="Re-train";
      document.getElementById("train-btn").disabled=false;
    }
    if(st.status==="error"){
      clearInterval(iv);
      lb.innerHTML+=`<p style="color:#e85d3a">❌ ${st.error}</p>`;
      document.getElementById("train-btn").innerHTML="Retry";
      document.getElementById("train-btn").disabled=false;
    }
  },1500);
}

function showMR(results,best,totalGames){
  document.getElementById("sbar").className="sbar ok";
  document.getElementById("sdot").classList.remove("pulse");
  document.getElementById("stext").textContent=`✓ Ready · ${totalGames} games · Best model: ${best}`;
  document.getElementById("mgrid").innerHTML=Object.entries(results).sort((a,b)=>b[1]-a[1]).map(([n,acc])=>`
    <div class="mc ${n===best?"best":""}">
      <div><div class="mcn">${n}</div>${n===best?"<div class=\"bbadge\">⭐ Best</div>":""}</div>
      <div class="mca" style="color:${acc>=70?"#2ecc71":acc>=55?"#c9a84c":"#e85d3a"}">${acc}%</div>
    </div>`).join("");
  document.getElementById("mresults").style.display="block";
  // Final chart with all results
  chartData={labels:Object.keys(results),values:Object.values(results)};
  drawChart();
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

function switchTab(btn, showId, hideId){
  document.getElementById(showId).style.display='block';
  document.getElementById(hideId).style.display='none';
  btn.parentElement.querySelectorAll('.otab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
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

  // Opening cards with boards + puzzles + chess.com link
  document.getElementById("olist").innerHTML=d.openings.map((o,i)=>{
    const bid=`board_${i}`;
    const puzzleHTML=(o.puzzles||[]).map((pz,pi)=>`
      <div class="pcard" id="pcard_${i}_${pi}">
        <div class="pnum">Puzzle ${pi+1}</div>
        <div class="pprompt">${pz.prompt}</div>
        <div class="phint" id="phint_${i}_${pi}" style="display:none">💡 ${pz.hint}</div>
        <div class="pans" id="pans_${i}_${pi}" style="display:none">✅ Answer: <strong>${pz.answer}</strong></div>
        <div class="pbtnrow">
          <button class="bbtn" onclick="document.getElementById('phint_${i}_${pi}').style.display='block'">Show Hint</button>
          <button class="bbtn" style="background:rgba(46,204,113,0.15);color:#2ecc71" onclick="document.getElementById('pans_${i}_${pi}').style.display='block'">Reveal Answer</button>
        </div>
      </div>`).join("");
    return `
    <div class="ocard" style="border-top:3px solid ${col}">
      <div class="oinfo">
        <div class="orank">#${i+1} RECOMMENDATION</div>
        <div class="oname">${o.name}</div>
        <div class="opgn">${o.pgn}</div>
        <div class="owhy">${o.why}</div>
        <div class="ometa"><strong>Difficulty:</strong> ${o.difficulty} &nbsp;·&nbsp; <strong>Famous players:</strong> ${o.players}</div>
        <div class="octa">
          <a href="${o.chesscom||'https://www.chess.com/learn-how-to-play-chess'}" target="_blank" class="btn btn-chesscom">♟ Try on Chess.com</a>
        </div>
      </div>
      <div class="bwrap">
        <div class="otabs">
          <button class="otab active" onclick="switchTab(this,'${bid}_board','${bid}_puzzles')">📋 Opening Moves</button>
          <button class="otab" onclick="switchTab(this,'${bid}_puzzles','${bid}_board')">🧩 Puzzles (${(o.puzzles||[]).length})</button>
        </div>
        <div id="${bid}_board">
          ${buildBoardHTML(bid)}
          <div class="mctr" id="${bid}_ctr"></div>
          <div class="bcontrols">
            <button class="bbtn" onclick="resetBoard('${bid}')">⏮ Start</button>
            <button class="bbtn" onclick="stepMove('${bid}',-1)">◀</button>
            <button class="bbtn" onclick="playAll('${bid}')">▶ Play</button>
            <button class="bbtn" onclick="stepMove('${bid}',1)">▶|</button>
          </div>
        </div>
        <div id="${bid}_puzzles" style="display:none">
          ${puzzleHTML||'<p style="color:var(--muted);padding:16px">No puzzles for this opening yet.</p>'}
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
    import os
    print("\n"+"="*50)
    print("  Chess Opening Recommender — GUI")
    print("="*50)
    print("\n  Open your browser at:  http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("="*50+"\n")
    port=int(os.environ.get("PORT",8080))
    app.run(debug=False,host="0.0.0.0",port=port)

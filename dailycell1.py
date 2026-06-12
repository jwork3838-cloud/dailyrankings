# =============================================================================
# INSIDER'S EDGE DAILY INFERENCE SCRIPT – PRODUCTION FINAL v15
# FIXED: P59 - Proper Bayesian Shrinkage for Pitcher Splits
# FIXED: P60 - Replaced 0.0 Pitcher Statcast defaults with League Averages
# FIXED: P61 - Added Missingness Indicators (bat_tracking_missing, pitcher_statcast_missing)
# FIXED: P62 - Added Z-Score Impact Explainer for Top 10 Predictions
# NEW:   P63 - DraftKings HR Props Integration → +EV Engine
#              Ports odds extraction from ScriptableDKodds.js (v50).
#              Fetches DK 1+/2+/3+ HR lines, fuzzy-matches to model players,
#              computes implied probability and edge for every player.
#              Adds DK_Odds, DK_Implied, Edge_Pct, Plus_EV columns to sheet.
# =============================================================================

import gspread
import pandas as pd
import numpy as np
import requests
import math
import time
import threading
import pickle
import os
import hashlib
import json
import functools
import re
import unicodedata
import urllib.parse
from datetime import datetime as dt_module, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO
import filelock

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# --- GITHUB ACTIONS / SERVICE ACCOUNT AUTHENTICATION ---
google_creds_json = os.environ.get("GOOGLE_CREDENTIALS")
if not google_creds_json:
    raise ValueError("GOOGLE_CREDENTIALS environment variable not found! Please add it to GitHub Secrets.")

creds_dict = json.loads(google_creds_json)
gc = gspread.service_account_from_dict(creds_dict)

SHEET_ID = "1-x26n6EhADNJgOdMHhJ4sz6XR52Qi1Hsgot1mbmClK8"
sh = gc.open_by_key(SHEET_ID)
training_ws = sh.worksheet("TrainingData")

TARGET_DATE = dt_module.now().strftime("%Y-%m-%d")
SEASON = int(TARGET_DATE.split('-')[0])
CACHE_DIR = "mlb_cache"
MODEL_DIR = "mlb_models"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
CACHE_TTL_HOURS = 24

FORCE_RETRAIN = True

MODEL_SCHEMA_VERSION = 7
FEATURE_COLS = [
    'park_factor_hr', 'temperature', 'wind_speed', 'wind_direction', 'wind_along_flight',
    'platoon_advantage',
    'batter_split_woba', 'batter_split_iso', 'batter_split_slg', 'batter_split_hr_rate', 'batter_split_pa',
    'pitcher_split_woba', 'pitcher_split_iso', 'pitcher_split_slg', 'pitcher_split_hr_rate', 'pitcher_split_pa',
    'batter_fb_pct', 'pitcher_hr_9',
    'pa', 'avg_hit_angle', 'sweetspot_percent', 'avg_hit_speed', 'ev50', 'barrel_rate',
    'est_ba', 'est_slg', 'est_woba', 'avg_bat_speed', 'hard_swing_rate', 'squared_up_per_swing',
    'hot_pa', 'hot_iso', 'hot_woba', 'hot_hr_rate', 'pitcher_avg_hit_angle', 'pitcher_avg_hit_speed',
    'pitcher_ev50', 'pitcher_barrel_rate', 'pitcher_xera', 'power_vs_power',
    'bat_tracking_missing', 'pitcher_statcast_missing'
]
FEATURE_VERSION = 7

DRIFT_SKIP_FEATURES = {
    'park_factor_hr', 'temperature', 'wind_speed', 'wind_direction', 'wind_along_flight'
}

parkFactorHR = {
    "ARI":1.12,"ATL":1.05,"BAL":0.98,"BOS":1.06,"CHC":0.95,"CWS":0.92,"CIN":1.23,
    "CLE":0.88,"COL":1.42,"DET":1.00,"HOU":1.09,"KC":1.02,"LAA":1.01,"LAD":1.08,
    "MIA":0.91,"MIL":1.10,"MIN":1.04,"NYM":0.96,"NYY":1.13,"ATH":0.97,"PHI":1.07,
    "PIT":0.94,"SD":0.89,"SF":0.85,"SEA":0.93,"STL":0.99,"TB":0.90,"TEX":1.11,
    "TOR":1.08,"WSH":1.02
}

stadiumData = {
    "Diamondbacks": {"lat":33.445,"lon":-112.067,"isDome":True, "abbrev":"ARI","teamId":109,"center_bearing":180},
    "Braves":       {"lat":33.891,"lon": -84.468,"isDome":False,"abbrev":"ATL","teamId":144,"center_bearing": 20},
    "Orioles":      {"lat":39.284,"lon": -76.622,"isDome":False,"abbrev":"BAL","teamId":110,"center_bearing": 85},
    "Red Sox":      {"lat":42.346,"lon": -71.097,"isDome":False,"abbrev":"BOS","teamId":111,"center_bearing": 40},
    "Cubs":         {"lat":41.948,"lon": -87.656,"isDome":False,"abbrev":"CHC","teamId":112,"center_bearing":105},
    "White Sox":    {"lat":41.830,"lon": -87.634,"isDome":False,"abbrev":"CWS","teamId":145,"center_bearing":100},
    "Reds":         {"lat":39.097,"lon": -84.507,"isDome":False,"abbrev":"CIN","teamId":113,"center_bearing": 15},
    "Guardians":    {"lat":41.496,"lon": -81.685,"isDome":False,"abbrev":"CLE","teamId":114,"center_bearing": 25},
    "Rockies":      {"lat":39.756,"lon":-104.994,"isDome":False,"abbrev":"COL","teamId":115,"center_bearing": 20},
    "Tigers":       {"lat":42.339,"lon": -83.049,"isDome":False,"abbrev":"DET","teamId":116,"center_bearing":100},
    "Astros":       {"lat":29.757,"lon": -95.356,"isDome":True, "abbrev":"HOU","teamId":117,"center_bearing":  0},
    "Royals":       {"lat":39.051,"lon": -94.480,"isDome":False,"abbrev":"KC", "teamId":118,"center_bearing": 80},
    "Angels":       {"lat":33.800,"lon":-117.883,"isDome":False,"abbrev":"LAA","teamId":108,"center_bearing": 10},
    "Dodgers":      {"lat":34.074,"lon":-118.240,"isDome":False,"abbrev":"LAD","teamId":119,"center_bearing":105},
    "Marlins":      {"lat":25.778,"lon": -80.220,"isDome":True, "abbrev":"MIA","teamId":146,"center_bearing":  0},
    "Brewers":      {"lat":43.028,"lon": -87.971,"isDome":True, "abbrev":"MIL","teamId":158,"center_bearing":  0},
    "Twins":        {"lat":44.982,"lon": -93.278,"isDome":False,"abbrev":"MIN","teamId":142,"center_bearing":105},
    "Mets":         {"lat":40.757,"lon": -73.846,"isDome":False,"abbrev":"NYM","teamId":121,"center_bearing":100},
    "Yankees":      {"lat":40.829,"lon": -73.926,"isDome":False,"abbrev":"NYY","teamId":147,"center_bearing":  5},
    "Athletics":    {"lat":38.580,"lon":-121.514,"isDome":False,"abbrev":"ATH","teamId":133,"center_bearing": 20},
    "Phillies":     {"lat":39.906,"lon": -75.167,"isDome":False,"abbrev":"PHI","teamId":143,"center_bearing":110},
    "Pirates":      {"lat":40.447,"lon": -80.006,"isDome":False,"abbrev":"PIT","teamId":134,"center_bearing": 10},
    "Padres":       {"lat":32.707,"lon":-117.157,"isDome":False,"abbrev":"SD", "teamId":135,"center_bearing":100},
    "Giants":       {"lat":37.779,"lon":-122.389,"isDome":False,"abbrev":"SF", "teamId":137,"center_bearing": 10},
    "Mariners":     {"lat":47.591,"lon":-122.333,"isDome":True, "abbrev":"SEA","teamId":136,"center_bearing":  0},
    "Cardinals":    {"lat":38.623,"lon": -90.193,"isDome":False,"abbrev":"STL","teamId":138,"center_bearing":100},
    "Rays":         {"lat":27.768,"lon": -82.653,"isDome":True, "abbrev":"TB", "teamId":139,"center_bearing":  0},
    "Rangers":      {"lat":32.751,"lon": -97.083,"isDome":True, "abbrev":"TEX","teamId":140,"center_bearing":  0},
    "Blue Jays":    {"lat":43.641,"lon": -79.390,"isDome":True, "abbrev":"TOR","teamId":141,"center_bearing":  0},
    "Nationals":    {"lat":38.873,"lon": -77.007,"isDome":False,"abbrev":"WSH","teamId":120,"center_bearing": 10},
}

TEAM_ABBREV = {
    "Arizona Diamondbacks":  "ARI",
    "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC",
    "Chicago White Sox":     "CWS",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KC",
    "Los Angeles Angels":    "LAA",
    "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA",
    "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN",
    "New York Mets":         "NYM",
    "New York Yankees":      "NYY",
    "Oakland Athletics":     "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SD",
    "San Francisco Giants":  "SF",
    "Seattle Mariners":      "SEA",
    "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TB",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Washington Nationals":  "WSH",
}

def abbrev_team(name):
    return TEAM_ABBREV.get(name, (name[:3].upper() if name else "???"))

DK_LEAGUE_ID        = "84240"
DK_HR_SUBCATEGORY   = "17319"               
DK_HR_LIVE_SUBCATS  = ["17482", "17320", "17321"]  
DK_SITE             = "US-PA-SB"

DK_HEADERS = {
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://sportsbook.draftkings.com",
    "Referer":         "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=games&subcategory=batter-props&nav_1=home-runs",
    "User-Agent":      "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
}

def _build_dk_url(sub_cat_id):
    ev_q = urllib.parse.quote(
        f"$filter=leagueId eq '{DK_LEAGUE_ID}' AND "
        f"clientMetadata/Subcategories/any(s: s/Id eq '{sub_cat_id}')"
    )
    mk_q = urllib.parse.quote(
        f"$filter=clientMetadata/subCategoryId eq '{sub_cat_id}'"
    )
    tv_q = urllib.parse.quote(f"{DK_LEAGUE_ID},{sub_cat_id}")
    return (
        f"https://sportsbook-nash.draftkings.com/sites/{DK_SITE}"
        f"/api/sportscontent/controldata/league/leagueSubcategory/v1/markets"
        f"?isBatchable=false&templateVars={tv_q}"
        f"&eventsQuery={ev_q}&marketsQuery={mk_q}&include=Events&entity=events"
    )

def _normalize_dk(name):
    if not name:
        return ""
    norm = unicodedata.normalize("NFD", name)
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = norm.lower()
    norm = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()'\"]+", "", norm)
    norm = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", norm)
    return re.sub(r"\s+", " ", norm).strip()

def _names_match_dk(n1, n2):
    if not n1 or not n2:
        return False
    a, b = _normalize_dk(n1), _normalize_dk(n2)
    if a == b:
        return True
    pa, pb = a.split(), b.split()
    if len(pa) >= 2 and len(pb) >= 2:
        if pa[-1] == pb[-1] and pa[0] and pb[0] and pa[0][0] == pb[0][0]:
            if pa[0] in pb[0] or pb[0] in pa[0]:
                return True
    return False

MIN_DK_PA = 50

def american_to_implied_prob(odds):
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)

def american_to_decimal(odds):
    if odds > 0:
        return odds / 100.0 + 1.0
    else:
        return 100.0 / abs(odds) + 1.0

def calc_ev_units(p_model, odds):
    decimal = american_to_decimal(odds)
    return round(p_model * decimal - 1.0, 4)

def calc_kelly(p_model, odds, cap=0.05):
    decimal = american_to_decimal(odds)
    b = decimal - 1.0
    if b <= 0:
        return 0.0
    kelly = (b * p_model - (1.0 - p_model)) / b
    return round(max(0.0, min(cap, kelly)), 4)

def fetch_dk_hr_props():
    all_cats = [DK_HR_SUBCATEGORY] + DK_HR_LIVE_SUBCATS
    player_odds = {}   

    for cat_id in all_cats:
        try:
            resp = requests.get(
                _build_dk_url(cat_id),
                headers=DK_HEADERS,
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[DK] Subcategory {cat_id} failed: {exc}")
            continue

        event_map = {}
        for ev in data.get("events") or []:
            home, away = ev.get("homeTeamName", ""), ev.get("awayTeamName", "")
            if not home and ev.get("participants"):
                for part in ev["participants"]:
                    role = part.get("venueRole", "")
                    pname = (
                        (part.get("metadata") or {}).get("shortName")
                        or (part.get("metadata") or {}).get("rosettaTeamName")
                        or part.get("name", "")
                    )
                    if role == "Home":
                        home = pname
                    elif role == "Away":
                        away = pname
            event_map[ev["id"]] = {"home": home, "away": away}

        for sel in data.get("selections") or []:
            label = (sel.get("label") or "").strip()
            if label not in ("1+", "2+", "3+"):
                continue

            p_info = next(
                (x for x in (sel.get("participants") or [])
                 if not x.get("type") or x.get("type") == "Player"),
                None
            )
            if not p_info:
                continue
            name = p_info.get("name") or p_info.get("displayName")
            if not name:
                continue

            odds_raw = None
            if sel.get("displayOdds"):
                odds_raw = sel["displayOdds"].get("american")
            if odds_raw is None:
                odds_raw = sel.get("americanOdds")
            if odds_raw is None:
                continue
            try:
                odds = float(str(odds_raw).replace("+", ""))
            except (ValueError, TypeError):
                continue

            if name not in player_odds:
                player_odds[name] = {}
            existing = player_odds[name].get(label)
            if existing is None or odds > existing:
                player_odds[name][label] = odds

    return player_odds

def attach_dk_ev(payload, dk_player_odds):
    batter_name = payload.get("batter_name", "")
    matched_name = next(
        (dk for dk in dk_player_odds if _names_match_dk(dk, batter_name)),
        None
    )

    payload["dk_line"]         = None
    payload["dk_odds"]         = None
    payload["dk_implied_prob"] = None
    payload["dk_edge"]         = None
    payload["ev_units"]        = None
    payload["kelly_pct"]       = None
    payload["dk_plus_ev"]      = False

    if not matched_name:
        return

    odds_dict = dk_player_odds[matched_name]
    for line in ("1+", "2+", "3+"):
        if line in odds_dict:
            payload["dk_line"] = line
            payload["dk_odds"] = odds_dict[line]
            break

    if payload["dk_odds"] is None:
        return

    p   = payload["hr_probability"]
    o   = payload["dk_odds"]
    pa  = payload.get("pa", 0) or 0

    imp   = american_to_implied_prob(o)
    ev    = calc_ev_units(p, o)
    kelly = calc_kelly(p, o)               

    payload["dk_implied_prob"] = round(imp * 100.0, 2)
    payload["dk_edge"]         = round((p - imp) * 100.0, 2)
    payload["ev_units"]        = ev
    payload["kelly_pct"]       = round(kelly * 100.0, 2)  
    payload["dk_plus_ev"]      = ev > 0 and pa >= MIN_DK_PA

thread_local = threading.local()
_all_sessions = []
_session_lock = threading.Lock()
NETWORK_WORKERS = 6
CPU_WORKERS = 1

def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=NETWORK_WORKERS, pool_maxsize=NETWORK_WORKERS)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        thread_local.session = s
        with _session_lock: _all_sessions.append(s)
    return thread_local.session

def fetch_json(url):
    try:
        r = get_session().get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        return r.json()
    except: return None

def fetch_csv(url):
    for attempt in range(3):
        try:
            r = get_session().get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            if 'html' in r.text.lower()[:500]: raise ValueError("HTML returned instead of CSV")
            df = pd.read_csv(StringIO(r.text))
            if df.empty: raise ValueError("Empty CSV")
            return df
        except:
            if attempt == 2: raise
            time.sleep(2 ** attempt)
    return None

def cache_key(prefix, *args):
    arg_str = '_'.join(str(a) for a in args)
    return os.path.join(CACHE_DIR, f"{prefix}_{hashlib.sha256(arg_str.encode()).hexdigest()[:16]}.pkl")

def is_cache_valid(filepath):
    if not os.path.exists(filepath): return False
    return dt_module.now() - dt_module.fromtimestamp(os.path.getmtime(filepath)) < timedelta(hours=CACHE_TTL_HOURS)

def get_cached_or_fetch(prefix, fetch_func, *args, **kwargs):
    cache_file = cache_key(prefix, *args)
    with filelock.FileLock(cache_file + ".lock"):
        if is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f: return pickle.load(f)
            except: pass
        data = fetch_func(*args, **kwargs)
        with open(cache_file, 'wb') as f: pickle.dump(data, f)
        return data

def _fetch_player_side(player_id, side_type):
    data = fetch_json(f"https://statsapi.mlb.com/api/v1/people/{player_id}")
    return data['people'][0].get(side_type, {}).get('code', 'R') if data and 'people' in data else 'R'

def get_player_side(player_id, side_type='batSide'):
    return get_cached_or_fetch(f"side_{side_type}", _fetch_player_side, str(player_id), side_type)

def _fetch_player_splits(pid, group, season):
    data = fetch_json(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=statSplits&group={group}&sitCodes=vl,vr&season={season}")
    empty = {'woba': 0.0, 'iso': 0.0, 'slg': 0.0, 'hr_rate': 0.0, 'pa': 0}
    splits = {'L': dict(empty), 'R': dict(empty)}
    if data and data.get('stats'):
        for stat_block in data['stats']:
            for split in stat_block.get('splits', []):
                code = split.get('split', {}).get('code')
                if code not in ('vl', 'vr'): continue
                hand_key = 'L' if code == 'vl' else 'R'
                s = split.get('stat', {})
                try:
                    pa  = int(s.get('plateAppearances', 0) or 0)
                    slg = float(s.get('slg', 0.0) or 0.0)
                    avg = float(s.get('avg', 0.0) or 0.0)
                    hr  = int(s.get('homeRuns', 0) or 0)
                    splits[hand_key] = {
                        'woba':    float(s.get('woba', 0.0) or 0.0),
                        'iso':     round(slg - avg, 4),
                        'slg':     slg,
                        'hr_rate': round(hr / pa, 4) if pa > 0 else 0.0,
                        'pa':      pa,
                    }
                except: pass
    return splits

def get_player_splits(pid, group, season):
    return get_cached_or_fetch(f"splits_{group}", _fetch_player_splits, str(pid), group, season)

def _fetch_player_gamelog(player_id, season):
    data = fetch_json(f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=hitting&season={season}")
    return data['stats'][0]['splits'] if data and data.get('stats') and data['stats'][0].get('splits') else []

def get_player_gamelog(player_id, season):
    return get_cached_or_fetch(f"gamelog", _fetch_player_gamelog, str(player_id), season)

def parse_innings_pitched(ip_val):
    try:
        s = str(ip_val).strip()
        if '.' in s:
            parts = s.split('.')
            return int(parts[0]) + int(parts[1]) / 3.0
        return float(s)
    except: return 0.0

def _fetch_pitcher_season_stats(pid, season):
    data = fetch_json(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season&group=pitching&season={season}")
    result = {'hr': 0, 'ip': 0.0}
    if data and data.get('stats') and data['stats'][0].get('splits'):
        s = data['stats'][0]['splits'][0].get('stat', {})
        result['hr'] = float(s.get('homeRuns', 0) or 0)
        result['ip'] = parse_innings_pitched(s.get('inningsPitched', 0))
    return result

def get_pitcher_season_stats(pid, season):
    return get_cached_or_fetch(f"pitcher_season", _fetch_pitcher_season_stats, str(pid), season)

@functools.lru_cache(maxsize=5000)
def get_hot_streak(player_id, target_date, games_back=7):
    logs = sorted([g for g in get_player_gamelog(player_id, SEASON) + get_player_gamelog(player_id, SEASON-1) if g.get('date') and g['date'] < target_date], key=lambda x: x['date'], reverse=True)[:games_back]
    pa = ab = singles = doubles = triples = hr = bb = hbp = 0
    for game in logs:
        s = game.get('stat', {})
        hits, dbl, tpl, hrs = s.get('hits',0), s.get('doubles',0), s.get('triples',0), s.get('homeRuns',0)
        pa += s.get('plateAppearances',0)
        ab += s.get('atBats',0)
        hr += hrs
        bb += s.get('baseOnBalls',0)
        hbp += s.get('hitByPitch',0) if 'hitByPitch' in s else s.get('hitByPound',0)
        singles += hits - dbl - tpl - hrs
        doubles += dbl
        triples += tpl
    if pa == 0: return {'hot_pa':0,'hot_iso':0,'hot_woba':0,'hot_hr_rate':0}
    iso = (doubles + triples*2 + hr*3) / max(ab,1)
    woba = (0.69*(bb+hbp) + 0.89*singles + 1.27*doubles + 1.62*triples + 2.10*hr) / pa
    return {'hot_pa':pa,'hot_iso':iso,'hot_woba':woba,'hot_hr_rate':hr/pa}

_weather_failures, _weather_total = 0, 0
def fetch_live_weather(stadium, date_str, game_time_utc=None):
    global _weather_failures, _weather_total
    _weather_total += 1
    if stadium.get('isDome'): return {'temp':70,'wind_speed':0,'wind_dir':0}
    lat, lon = stadium['lat'], stadium['lon']
    data = fetch_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m,wind_direction_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=UTC&start_date={date_str}&end_date={date_str}")
    try:
        if data and 'hourly' in data:
            times, temps, winds, dirs = data['hourly']['time'], data['hourly']['temperature_2m'], data['hourly']['wind_speed_10m'], data['hourly']['wind_direction_10m']
            if game_time_utc:
                game_dt = dt_module.strptime(game_time_utc, "%Y-%m-%dT%H:%M:%SZ")
                if game_dt.minute >= 30: game_dt += timedelta(hours=1)
                target = game_dt.replace(minute=0,second=0).strftime("%Y-%m-%dT%H:00")
                if target in times:
                    idx = times.index(target)
                    if idx < len(temps) and temps[idx] is not None:
                        return {'temp':temps[idx], 'wind_speed':winds[idx] or 0, 'wind_dir':dirs[idx] or 0}
            valid_temps = [t for t in temps if t is not None]
            if valid_temps: return {'temp':np.median(valid_temps), 'wind_speed':np.median([w for w in winds if w is not None]) or 0, 'wind_dir':np.median([d for d in dirs if d is not None]) or 0}
    except: pass
    _weather_failures += 1
    return {'temp':70,'wind_speed':0,'wind_dir':0}

def get_live_team_roster(team_id):
    data = fetch_json(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active")
    return [{'id': str(p['person']['id']), 'name': p['person']['fullName']} for p in data.get('roster', []) if p.get('position',{}).get('abbreviation') != 'P'] if data else []

COLUMN_ALIASES = {
    'brl_percent':   ['brl_percent', 'barrel_batted_rate', 'barrel_pct', 'barrel_rate'],
    'avg_hit_speed': ['avg_hit_speed', 'launch_speed', 'exit_velocity', 'avg_exit_velocity'],
    'fb_percent':    ['fb_percent', 'fly_ball_percent', 'fb_pct', 'flyballs_percent', 'fly_ball_rate', 'flyball_percent', 'flyball_rate', 'fb_rate', 'flyballs_percent_pa', 'fly_ball_pct'],
    'pa':            ['attempts', 'pa', 'plate_appearances', 'n_pa', 'total_pa', 'num_pa', 'pas', 'pa_total'],
    'avg_hit_angle': ['avg_hit_angle', 'launch_angle', 'avg_launch_angle'],
    'b_home_run':    ['b_home_run', 'home_run', 'hr', 'home_runs', 'pitcher_home_runs'],
    'outs_pitched':  ['outs_pitched', 'outs', 'total_outs', 'p_outs'],
    'sweetspot_percent': ['anglesweetspotpercent', 'sweet_spot_percent', 'sweetspot_percent']
}

def resolve_column(df, logical_name):
    for alias in COLUMN_ALIASES.get(logical_name, [logical_name]):
        if alias in df.columns: return alias
    return None

def validate_endpoint_soft(df, name, critical_columns):
    missing_hard, missing_soft = [], []
    for col in critical_columns:
        if resolve_column(df, col) is None:
            if col in {'player_id', 'playerid', 'id'}: missing_hard.append(col)
            else: missing_soft.append(col)
    if missing_soft: print(f"[FINGERPRINT WARNING] {name}: columns not found (will use fallback defaults): {missing_soft}")
    if missing_hard: raise RuntimeError(f"Endpoint {name} missing player ID column. Tried: {missing_hard}. Actual columns: {list(df.columns)}")
    print(f"[FINGERPRINT] {name}: OK (player_id present, {len(missing_soft)} soft columns missing)")

def load_statcast_data(season):
    for yr in [season] + [season - i for i in range(1,5)]:
        print(f"Attempting Statcast baseline for {yr}...")
        try:
            df_b = fetch_csv(f"https://baseballsavant.mlb.com/statcast_leaderboard?type=batter&year={yr}&position=&team=&min=1&csv=true")
            if df_b is None or df_b.empty: continue
            validate_endpoint_soft(df_b, 'batter_leaderboard', ['pa', 'avg_hit_speed', 'brl_percent', 'fb_percent', 'player_id'])
            df_p = fetch_csv(f"https://baseballsavant.mlb.com/statcast_leaderboard?type=pitcher&year={yr}&position=&team=&min=1&csv=true")
            validate_endpoint_soft(df_p, 'pitcher_leaderboard', ['avg_hit_speed', 'player_id'])
            df_exp = fetch_csv(f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={yr}&min=1&csv=true")
            df_bt = fetch_csv(f"https://baseballsavant.mlb.com/leaderboard/bat-tracking?year={yr}&min=1&csv=true")
            df_pe = fetch_csv(f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year={yr}&min=1&csv=true")

            if all(df is not None and not df.empty for df in [df_b, df_p]):
                print(f"Successfully loaded {yr} data.")
                id_cols = ['player_id','playerid','id']
                b_col = next((c for c in id_cols if c in df_b.columns), None)
                p_col = next((c for c in id_cols if c in df_p.columns), None)
                exp_col = next((c for c in id_cols if c in df_exp.columns), None) if df_exp is not None else None
                bt_col = next((c for c in id_cols if c in df_bt.columns), None) if df_bt is not None else None
                pe_col = next((c for c in id_cols if c in df_pe.columns), None) if df_pe is not None else None

                return {
                    'batters': {str(k): v for k, v in df_b.set_index(b_col).to_dict('index').items()},
                    'pitchers': {str(k): v for k, v in df_p.set_index(p_col).to_dict('index').items()},
                    'expected': {str(k): v for k, v in df_exp.set_index(exp_col).to_dict('index').items()} if df_exp is not None else {},
                    'bat_track': {str(k): v for k, v in df_bt.set_index(bt_col).to_dict('index').items()} if df_bt is not None else {},
                    'pitcher_exp': {str(k): v for k, v in df_pe.set_index(pe_col).to_dict('index').items()} if df_pe is not None else {},
                    'year': yr,
                    'league_avg': {
                        'batter': df_b.select_dtypes(include=[np.number]).median().to_dict() if not df_b.empty else {},
                        'pitcher': df_p.select_dtypes(include=[np.number]).median().to_dict() if not df_p.empty else {}
                    }
                }
        except Exception as e:
            print(f"Year {yr} failed: {e}")
            continue
    raise RuntimeError("Could not load Statcast data for any recent season.")

def safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
        f = float(val)
        return f if np.isfinite(f) else default
    except: return default

def safe_get(row, keys, default=0.0):
    if row is None:
        return default
    _is_numeric = isinstance(default, (int, float)) and default is not None
    for key in keys:
        if key in row:
            value = row[key]
            try:
                if pd.isna(value):
                    return default
            except Exception:
                pass
            if _is_numeric:
                try:
                    f = float(value)
                    return f if np.isfinite(f) else default
                except (TypeError, ValueError):
                    return default
            return value
    return default

def extract_metric(row, keys, default=0.0, context='', fallback_tracker=None, col_name=None):
    if row is None:
        if fallback_tracker and col_name: fallback_tracker[col_name] += 1
        return default, True
    for k in keys:
        if k in row and pd.notna(row[k]):
            try:
                val = float(row[k])
                if np.isfinite(val): return val, False
            except: continue
    if fallback_tracker and col_name: fallback_tracker[col_name] += 1
    return default, True

_feature_fallbacks = {col: 0 for col in FEATURE_COLS}
_total_feature_requests = 0
_fallbacks_lock = threading.Lock()

def shrink(metric, pa, prior_metric, prior_pa):
    return (metric * pa + prior_metric * prior_pa) / (pa + prior_pa)

def build_feature_vector(bid, pitcher_id, pitcher_hand, b_stat, p_stat, b_exp, b_track, p_exp, park_factor, weather, wind_along_flight, baseline_year, league_avg, hot):
    global _total_feature_requests
    with _fallbacks_lock: _total_feature_requests += 1

    hand = get_player_side(bid, 'batSide')
    if hand not in ('L','R','S'): hand = 'R'
    platoon = 1 if hand == 'S' or (hand=='L' and pitcher_hand=='R') or (hand=='R' and pitcher_hand=='L') else 0

    b_splits = get_player_splits(bid, 'hitting', baseline_year)
    p_splits = get_player_splits(pitcher_id, 'pitching', baseline_year)
    batter_side = ('L' if pitcher_hand=='R' else 'R') if hand=='S' else hand

    b_split_vs = b_splits.get(pitcher_hand, {})
    p_split_vs = p_splits.get(batter_side, {})

    b_split_pa = b_split_vs.get('pa', 0)
    batter_split_woba    = shrink(b_split_vs.get('woba', 0.320), b_split_pa, 0.320, 250)
    batter_split_iso     = shrink(b_split_vs.get('iso', 0.150), b_split_pa, 0.150, 250)
    batter_split_slg     = shrink(b_split_vs.get('slg', 0.420), b_split_pa, 0.420, 250)
    batter_split_hr_rate = shrink(b_split_vs.get('hr_rate', 0.030), b_split_pa, 0.030, 250)

    p_split_pa = p_split_vs.get('pa', 0)
    pitcher_split_woba    = shrink(p_split_vs.get('woba', 0.320), p_split_pa, 0.320, 250)
    pitcher_split_iso     = shrink(p_split_vs.get('iso', 0.150), p_split_pa, 0.150, 250)
    pitcher_split_slg     = shrink(p_split_vs.get('slg', 0.420), p_split_pa, 0.420, 250)
    pitcher_split_hr_rate = shrink(p_split_vs.get('hr_rate', 0.030), p_split_pa, 0.030, 250)

    if b_stat is None: b_stat = league_avg['batter']
    if p_stat is None: p_stat = league_avg['pitcher']

    p_season = get_pitcher_season_stats(pitcher_id, baseline_year)
    if p_season['ip'] > 0:
        pitcher_hr_9 = (p_season['hr'] / p_season['ip']) * 9
    else:
        p_hr, _ = extract_metric(p_stat, ['b_home_run', 'home_run', 'hr'], 0, 'pitcher_hr', _feature_fallbacks, 'pitcher_hr_9')
        p_outs, _ = extract_metric(p_stat, ['outs_pitched', 'outs', 'total_outs', 'p_outs'], 0, 'pitcher_outs', _feature_fallbacks, 'pitcher_hr_9')
        pitcher_hr_9 = (p_hr / p_outs) * 27 if p_outs > 0 else 0.0

    def compute_fb_pct(stat_dict):
        if stat_dict is None: return 35.0
        fbld = safe_float(stat_dict.get('fbld', stat_dict.get('fly_balls', 0)))
        gb   = safe_float(stat_dict.get('gb',   stat_dict.get('ground_balls', 0)))
        if (fbld + gb) > 0: return round((fbld / (fbld + gb)) * 100, 2)
        for col in ['fb_percent','fly_ball_percent','fb_pct','flyballs_percent','fly_ball_rate']:
            if col in stat_dict and pd.notna(stat_dict[col]):
                v = safe_float(stat_dict[col])
                if v > 0: return v
        return 35.0

    def get_metric(stat_dict, logical_keys, default, col_name):
        actual_keys = []
        for lk in logical_keys: actual_keys.extend(COLUMN_ALIASES.get(lk, [lk]))
        val, _ = extract_metric(stat_dict, actual_keys, default, col_name, _feature_fallbacks, col_name)
        return val

    hot_pa = hot['hot_pa'] if hot else 0
    hot_woba = shrink(hot['hot_woba'], hot_pa, 0.320, 40) if hot else 0.320
    hot_iso = shrink(hot['hot_iso'], hot_pa, 0.150, 40) if hot else 0.150
    hot_hr_rate = shrink(hot['hot_hr_rate'], hot_pa, 0.030, 40) if hot else 0.030

    p_ahs_def = league_avg['pitcher'].get('avg_hit_speed', 88.5)
    p_ev50_def = league_avg['pitcher'].get('ev50', 95.0)
    p_brl_def = league_avg['pitcher'].get('brl_percent', 6.5)

    p_ahs = safe_get(p_stat, ['avg_hit_speed'], None)
    p_ev50 = safe_get(p_stat, ['ev50'], None)
    p_brl = safe_get(p_stat, ['brl_percent', 'barrel_batted_rate'], None)

    pitcher_statcast_missing = 1 if (p_ahs is None and p_ev50 is None) else 0
    bat_tracking_missing = 1 if safe_get(b_track, ['avg_bat_speed'], None) is None else 0

    vec = {
        'park_factor_hr': park_factor,
        'temperature': weather['temp'],
        'wind_speed': weather['wind_speed'],
        'wind_direction': weather['wind_dir'],
        'wind_along_flight': wind_along_flight,
        'platoon_advantage': platoon,
        'batter_split_woba':     batter_split_woba,
        'batter_split_iso':      batter_split_iso,
        'batter_split_slg':      batter_split_slg,
        'batter_split_hr_rate':  batter_split_hr_rate,
        'batter_split_pa':       b_split_pa,
        'pitcher_split_woba':    pitcher_split_woba,
        'pitcher_split_iso':     pitcher_split_iso,
        'pitcher_split_slg':     pitcher_split_slg,
        'pitcher_split_hr_rate': pitcher_split_hr_rate,
        'pitcher_split_pa':      p_split_pa,
        'batter_fb_pct': compute_fb_pct(b_stat),
        'pitcher_hr_9': pitcher_hr_9,
        'pa': get_metric(b_stat, ['pa'], 0, 'pa'),
        'avg_hit_angle': get_metric(b_stat, ['avg_hit_angle'], 0.0, 'avg_hit_angle'),
        'sweetspot_percent': get_metric(b_stat, ['sweetspot_percent'], 0.0, 'sweetspot_percent'),
        'avg_hit_speed': get_metric(b_stat, ['avg_hit_speed'], 0.0, 'avg_hit_speed'),
        'ev50': get_metric(b_stat, ['ev50'], 0.0, 'ev50'),
        'barrel_rate': get_metric(b_stat, ['brl_percent'], 0.0, 'barrel_rate'),
        'est_ba': get_metric(b_exp, ['est_ba', 'xba'], 0.250, 'est_ba'),
        'est_slg': get_metric(b_exp, ['est_slg', 'xslg'], 0.420, 'est_slg'),
        'est_woba': get_metric(b_exp, ['est_woba', 'xwoba'], 0.320, 'est_woba'),
        'avg_bat_speed': get_metric(b_track, ['avg_bat_speed'], 72.0, 'avg_bat_speed'),
        'hard_swing_rate': get_metric(b_track, ['hard_swing_rate'], 0.0, 'hard_swing_rate'),
        'squared_up_per_swing': get_metric(b_track, ['squared_up_per_swing'], 0.0, 'squared_up_per_swing'),
        'hot_pa': hot_pa,
        'hot_iso': hot_iso,
        'hot_woba': hot_woba,
        'hot_hr_rate': hot_hr_rate,
        'pitcher_avg_hit_angle': get_metric(p_stat, ['avg_hit_angle'], 0.0, 'pitcher_avg_hit_angle'),
        'pitcher_avg_hit_speed': p_ahs if p_ahs is not None else p_ahs_def,
        'pitcher_ev50':          p_ev50 if p_ev50 is not None else p_ev50_def,
        'pitcher_barrel_rate':   p_brl if p_brl is not None else p_brl_def,
        'pitcher_xera': get_metric(p_exp, ['xera', 'est_era'], 4.00, 'pitcher_xera'),
        'power_vs_power': 1 if (get_metric(b_track, ['avg_bat_speed'], 72.0, 'avg_bat_speed') > 75 and get_metric(p_stat, ['avg_hit_speed'], 88.0, 'pitcher_avg_hit_speed') < 86) else 0,
        'bat_tracking_missing': bat_tracking_missing,
        'pitcher_statcast_missing': pitcher_statcast_missing
    }
    for k in vec:
        if not np.isfinite(safe_float(vec[k])): vec[k] = 0.0
    return vec

def calculate_psi(expected, actual, bins=10):
    percentiles = np.linspace(0, 100, bins+1)
    bin_edges = np.percentile(expected, percentiles)
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 3: return 0.0
    bin_edges = unique_edges
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    expected_prop = np.clip(np.histogram(expected, bins=bin_edges)[0] / len(expected), 1e-8, 1)
    actual_prop = np.clip(np.histogram(actual, bins=bin_edges)[0] / len(actual), 1e-8, 1)
    return np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))

def calculate_ks(expected, actual):
    from scipy.stats import ks_2samp
    return ks_2samp(expected, actual).statistic

def monitor_drift(train_df, infer_df, feature_cols, train_stats):
    print("[DRIFT MONITOR] Computing PSI and KS...")
    if len(infer_df) < 100:
        print(f"  [NOTE] Inference set has {len(infer_df)} rows — PSI unreliable for slate-dependent features; those are skipped automatically.")
    for col in feature_cols:
        if col in DRIFT_SKIP_FEATURES: continue
        train_vals = train_df[col].dropna().values
        infer_vals = infer_df[col].dropna().values
        if len(train_vals) < 20 or len(infer_vals) < 20: continue
        psi = calculate_psi(train_vals, infer_vals, bins=10)
        if psi > 0.3: print(f"  CRITICAL {col}: PSI={psi:.3f} (>0.3)")
        elif psi > 0.2: print(f"  WARNING  {col}: PSI={psi:.3f} (>0.2)")
        ks = calculate_ks(train_vals, infer_vals)
        if ks > 0.3: print(f"  WARNING  {col}: KS={ks:.3f}")
        train_mean = train_stats['mean'][col]
        train_std  = max(train_stats['std'][col], 0.05)
        infer_mean = infer_vals.mean()
        z = abs(infer_mean - train_mean) / train_std
        if z > 3.0: print(f"  WARNING  {col}: mean shift z={z:.2f} (infer={infer_mean:.3f}, train={train_mean:.3f})")

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1]) if i == n_bins-1 else (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if not in_bin.any(): continue
        ece += (in_bin.sum() / len(y_true)) * abs(y_prob[in_bin].mean() - y_true[in_bin].mean())
    return ece

def get_dataset_hash(df):
    return hashlib.sha256(df.sort_values(by=list(df.columns)).reset_index(drop=True).to_csv(index=False).encode()).hexdigest()

MODEL_METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")
RF_MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_MODEL_FILE = os.path.join(MODEL_DIR, "xgb_model.pkl")

def save_models(rf_model, xgb_cal, metadata):
    with open(RF_MODEL_FILE, 'wb') as f: pickle.dump(rf_model, f)
    with open(XGB_MODEL_FILE, 'wb') as f: pickle.dump(xgb_cal, f)
    with open(MODEL_METADATA_FILE, 'w') as f: json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    print(f"[MODEL] Saved to {MODEL_DIR}")

def load_models():
    if not os.path.exists(RF_MODEL_FILE) or not os.path.exists(XGB_MODEL_FILE) or not os.path.exists(MODEL_METADATA_FILE): return None, None, None
    with open(RF_MODEL_FILE, 'rb') as f: rf_model = pickle.load(f)
    with open(XGB_MODEL_FILE, 'rb') as f: xgb_cal = pickle.load(f)
    with open(MODEL_METADATA_FILE, 'r') as f: metadata = json.load(f)
    return rf_model, xgb_cal, metadata

print("[PHASE 1] Loading training data from Google Sheets...")
raw_data = training_ws.get_all_records()
df_train = pd.DataFrame(raw_data)
if df_train.empty: raise ValueError("TrainingData sheet empty. Run backfill first.")

def normalize_actual_hr(series):
    mapping = {"1899-12-30":0,"1899-12-31":1,"0":0,"1":1,"False":0,"True":1,"FALSE":0,"TRUE":1}
    return series.astype(str).str.strip().str.replace("T00:00:00", "", regex=False).map(mapping).fillna(pd.to_numeric(series, errors="coerce")).fillna(0).astype(int)

df_train['actual_hr'] = normalize_actual_hr(df_train['actual_hr'])

current_dataset_hash = get_dataset_hash(df_train[FEATURE_COLS + ['actual_hr']])
latest_game_date = df_train['hr_training_date'].max() if 'hr_training_date' in df_train.columns else None
current_row_count = len(df_train)
current_positive_count = df_train['actual_hr'].sum()
current_positive_rate = current_positive_count / current_row_count if current_row_count > 0 else 0

rf_model, xgb_cal, saved_meta = load_models()
retrain_needed = False

if FORCE_RETRAIN:
    print("[MODEL] FORCE_RETRAIN=True — skipping saved model, retraining from scratch.")
    retrain_needed = True
elif saved_meta is not None:
    print(f"[MODEL] Saved model from {saved_meta.get('generated_at', 'unknown')}")
    if saved_meta.get('dataset_hash') != current_dataset_hash: retrain_needed = True
    elif saved_meta.get('row_count') != current_row_count or saved_meta.get('positive_count') != current_positive_count: retrain_needed = True
    elif saved_meta.get('positive_rate', 0) != current_positive_rate: retrain_needed = True
    elif latest_game_date and saved_meta.get('latest_game_date') and latest_game_date > saved_meta['latest_game_date']: retrain_needed = True
    elif saved_meta.get('schema_version') != MODEL_SCHEMA_VERSION: retrain_needed = True
    elif saved_meta.get('feature_version') != FEATURE_VERSION: retrain_needed = True
    else: print("[MODEL] Using saved models.")
else: retrain_needed = True

if retrain_needed:
    missing_cols = [c for c in FEATURE_COLS + ['actual_hr'] if c not in df_train.columns]
    if missing_cols: raise ValueError(f"Missing columns: {missing_cols}")

    for col in FEATURE_COLS: df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0.0)

    if 'hr_training_date' in df_train.columns:
        df_train = df_train.sort_values('hr_training_date').reset_index(drop=True)

    X_train, y_train = df_train[FEATURE_COLS], df_train['actual_hr']
    train_stats = {'mean': X_train.mean().to_dict(), 'std': X_train.std().to_dict(), 'median': X_train.median().to_dict()}

    MIN_ROWS, MIN_POSITIVE = max(500, len(FEATURE_COLS) * 20), max(10, int(max(500, len(FEATURE_COLS) * 20) * 0.005))
    if len(df_train) < MIN_ROWS: raise ValueError(f"Training set too small: {len(df_train)} rows, need {MIN_ROWS}")
    if int(y_train.sum()) < MIN_POSITIVE: raise ValueError(f"Too few positive HR examples: {int(y_train.sum())}, need {MIN_POSITIVE}")
    print(f"[AUDIT] Training set: {len(df_train)} rows, HR rate = {y_train.mean():.2%}")

    print("[PHASE 2] Training models...")
    cv = TimeSeriesSplit(n_splits=5)

    def get_calibrated_oof(base_estimator, X, y, cv):
        oof_probs = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X, y):
            X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
            y_train_f = y.iloc[train_idx]
            temp_cal = CalibratedClassifierCV(base_estimator, method='isotonic', cv=3)
            temp_cal.fit(X_train_f, y_train_f)
            oof_probs[val_idx] = temp_cal.predict_proba(X_val_f)[:,1]
        return oof_probs

    xgb_base = XGBClassifier(n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.85, colsample_bytree=0.85, min_child_weight=5, gamma=0.5, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=(y_train==0).sum() / max((y_train==1).sum(),1), random_state=42, n_jobs=CPU_WORKERS)
    xgb_cal = CalibratedClassifierCV(xgb_base, method='isotonic', cv=5)
    xgb_cal.fit(X_train, y_train)
    xgb_oof = get_calibrated_oof(xgb_base, X_train, y_train, cv)

    rf_base = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=CPU_WORKERS)
    rf_model = CalibratedClassifierCV(rf_base, method='isotonic', cv=5)
    rf_model.fit(X_train, y_train)
    rf_oof = get_calibrated_oof(rf_base, X_train, y_train, cv)

    xgb_auc, rf_auc = roc_auc_score(y_train, xgb_oof), roc_auc_score(y_train, rf_oof)
    rf_weight, xgb_weight = rf_auc / (rf_auc + xgb_auc), xgb_auc / (rf_auc + xgb_auc)
    print(f"RF Calibrated OOF AUC = {rf_auc:.4f}, XGB Calibrated OOF AUC = {xgb_auc:.4f}")
    print(f"Ensemble weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}")

    brier, ece = brier_score_loss(y_train, xgb_oof), expected_calibration_error(y_train, xgb_oof)
    print(f"[CALIBRATION] XGB OOF: Brier={brier:.4f}, ECE={ece:.4f}")

    metadata = {
        'generated_at': dt_module.now().isoformat(), 'season': SEASON, 'baseline_statcast_year': SEASON - 1,
        'row_count': len(df_train), 'positive_count': int(y_train.sum()), 'positive_rate': float(y_train.mean()),
        'latest_game_date': latest_game_date if latest_game_date else '', 'dataset_hash': current_dataset_hash,
        'schema_version': MODEL_SCHEMA_VERSION, 'feature_version': FEATURE_VERSION, 'train_stats': train_stats,
        'rf_weight': rf_weight, 'xgb_weight': xgb_weight, 'rf_auc': rf_auc, 'xgb_auc': xgb_auc,
        'calibration_brier': brier, 'calibration_ece': ece
    }
    save_models(rf_model, xgb_cal, metadata)
else:
    with open(MODEL_METADATA_FILE, 'r') as f: metadata = json.load(f)
    rf_weight, xgb_weight, train_stats = metadata['rf_weight'], metadata['xgb_weight'], metadata['train_stats']
    for col in FEATURE_COLS: df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0.0)
    X_train = df_train[FEATURE_COLS]

print(f"[PHASE 3] Fetching schedule for {TARGET_DATE}...")
schedule_data = fetch_json(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={TARGET_DATE}&endDate={TARGET_DATE}&hydrate=probablePitcher")
if not schedule_data or not schedule_data.get('dates'):
    print("No games scheduled.")
    exit(0)

games = schedule_data['dates'][0].get('games', [])
print(f"Found {len(games)} games.")

print("\n[PHASE 3.5] Fetching DraftKings HR prop lines...")
dk_player_odds = fetch_dk_hr_props()
dk_match_count  = len(dk_player_odds)
print(f"[DK] Lines fetched for {dk_match_count} players "
      f"({sum(1 for v in dk_player_odds.values() if '1+' in v)} with 1+ odds).")
if dk_match_count == 0:
    print("[DK] ⚠  No DK odds returned — edge column will be empty. "
          "Check if DK subcategory IDs have rotated or the API is geo-restricted.")

sc_data = load_statcast_data(SEASON - 1)
baseline_year, batters_dict, pitchers_dict, expected_dict, tracking_dict, pitcher_exp_dict, league_avg = sc_data['year'], sc_data['batters'], sc_data['pitchers'], sc_data['expected'], sc_data['bat_track'], sc_data['pitcher_exp'], sc_data['league_avg']
print(f"[STATCAST] Using baseline year {baseline_year} for inference features (matches backfill).")

all_pitcher_ids = set(str(game['teams'][side].get('probablePitcher', {}).get('id')) for game in games for side in ['away','home'] if game['teams'][side].get('probablePitcher', {}).get('id'))
with ThreadPoolExecutor(max_workers=NETWORK_WORKERS) as net_exec:
    futures = [net_exec.submit(get_player_side, pid, 'pitchHand') for pid in all_pitcher_ids] + [net_exec.submit(get_player_splits, pid, 'pitching', baseline_year) for pid in all_pitcher_ids]
    for f in futures: f.result()

inference_payloads, seen_matchups, matchup_lock, all_vectors = [], set(), threading.Lock(), []

for game in games:
    game_pk, game_time_utc, home_team_id = game['gamePk'], game.get('gameDate'), game['teams']['home']['team']['id']
    stadium = stadiumData.get(next((k for k,v in stadiumData.items() if v['teamId']==home_team_id), None))
    if not stadium: continue
    home_abbrev, park_factor = stadium['abbrev'], parkFactorHR.get(stadium['abbrev'], 1.0)
    weather = fetch_live_weather(stadium, TARGET_DATE, game_time_utc)
    wind_along_flight = weather['wind_speed'] * math.cos(math.radians(weather['wind_dir'] - stadium.get('center_bearing', 0) + 180))

    away_pitcher, home_pitcher = game['teams']['away'].get('probablePitcher', {}), game['teams']['home'].get('probablePitcher', {})
    away_pid, home_pid = str(away_pitcher.get('id')) if away_pitcher.get('id') else None, str(home_pitcher.get('id')) if home_pitcher.get('id') else None
    if not away_pid or not home_pid: continue
    away_p_name, home_p_name = away_pitcher.get('fullName','Unknown'), home_pitcher.get('fullName','Unknown')
    away_p_hand, home_p_hand = get_player_side(away_pid, 'pitchHand'), get_player_side(home_pid, 'pitchHand')

    box_data = fetch_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
    away_batters, home_batters = [], []
    if box_data:
        try:
            teams_box = box_data['liveData']['boxscore']['teams']
            def _is_original_starter(p):
                try:
                    order = int(p.get('battingOrder', 0))
                    return order % 100 == 0 and 100 <= order <= 900
                except: return False
            away_batters = [{'id': str(pid.replace('ID','')), 'name': p['person']['fullName']} for pid,p in teams_box['away']['players'].items() if _is_original_starter(p)]
            home_batters = [{'id': str(pid.replace('ID','')), 'name': p['person']['fullName']} for pid,p in teams_box['home']['players'].items() if _is_original_starter(p)]
        except: pass
    if not away_batters: away_batters = get_live_team_roster(game['teams']['away']['team']['id'])
    if not home_batters: home_batters = get_live_team_roster(game['teams']['home']['team']['id'])

    all_batter_ids = set(b['id'] for b in away_batters+home_batters)
    with ThreadPoolExecutor(max_workers=NETWORK_WORKERS) as net_exec:
        futures = [net_exec.submit(get_player_side, bid, 'batSide') for bid in all_batter_ids] + [net_exec.submit(get_player_splits, bid, 'hitting', baseline_year) for bid in all_batter_ids] + [net_exec.submit(get_hot_streak, bid, TARGET_DATE) for bid in all_batter_ids]
        for f in futures: f.result()

    away_team = game['teams']['away']['team']['name']
    home_team = game['teams']['home']['team']['name']

    for batter, pid, p_hand, p_name, team_name in (
        [(b, home_pid, home_p_hand, home_p_name, away_team) for b in away_batters] +
        [(b, away_pid, away_p_hand, away_p_name, home_team) for b in home_batters]
    ):
        bid = batter['id']
        with matchup_lock:
            if (bid, pid) in seen_matchups: continue
            seen_matchups.add((bid, pid))
        vec_dict = build_feature_vector(bid, pid, p_hand, batters_dict.get(bid), pitchers_dict.get(pid), expected_dict.get(bid, {}), tracking_dict.get(bid, {}), pitcher_exp_dict.get(pid, {}), park_factor, weather, wind_along_flight, baseline_year, league_avg, get_hot_streak(bid, TARGET_DATE))
        vec = [safe_float(vec_dict[col]) for col in FEATURE_COLS]
        all_vectors.append(vec)
        inference_payloads.append({
            'batter_name': batter.get('name', 'Unknown'),
            'team': abbrev_team(team_name),   
            'pitcher_name': p_name,
            'stadium': home_abbrev,
            'pa': safe_float(vec_dict.get('pa', 0)),   
            'vector': vec,
        })

if all_vectors: monitor_drift(X_train, pd.DataFrame(all_vectors, columns=FEATURE_COLS), FEATURE_COLS, train_stats)

fallback_rate = sum(_feature_fallbacks.values()) / max(_total_feature_requests * len(FEATURE_COLS), 1)
print(f"[FALLBACK RATE] Feature-level fallback: {fallback_rate:.2%}")
if fallback_rate > 0.25: raise RuntimeError(f"Excessive feature fallback rate ({fallback_rate:.2%}) – Statcast schema likely changed.")
if _weather_failures / max(_weather_total,1) > 0.5: print(f"[WARNING] Weather API failing often, continuing with fallback values.")

if not inference_payloads: print("No valid matchups.")
else:
    matrix_df = pd.DataFrame([p['vector'] for p in inference_payloads], columns=FEATURE_COLS)
    rf_probs, xgb_probs = rf_model.predict_proba(matrix_df)[:,1], xgb_cal.predict_proba(matrix_df)[:,1]
    ensemble_probs = rf_weight * rf_probs + xgb_weight * xgb_probs

    print(f"[PREDICTION SANITY] std={np.std(ensemble_probs):.4f}, max={np.max(ensemble_probs):.4f}, unique={len(np.unique(ensemble_probs))}")
    if np.std(ensemble_probs) < 0.01 or np.max(ensemble_probs) < 0.01: raise RuntimeError("Prediction collapse detected.")

    xgb_est = xgb_cal.calibrated_classifiers_[0].estimator
    importances = xgb_est.feature_importances_
    train_means = np.array(list(train_stats['mean'].values()))
    train_stds = np.array(list(train_stats['std'].values()))
    train_stds = np.where(train_stds == 0, 1e-9, train_stds) 

    for i, p in enumerate(inference_payloads):
        p['hr_probability'] = ensemble_probs[i]
        z_scores = (np.array(p['vector']) - train_means) / train_stds
        impacts = z_scores * importances
        impact_series = pd.Series(impacts, index=FEATURE_COLS)
        p['top_pos'] = impact_series.nlargest(3)
        p['top_neg'] = impact_series.nsmallest(3)
        attach_dk_ev(p, dk_player_odds)

    df_results = pd.DataFrame(inference_payloads)
    df_results['HR %'] = (df_results['hr_probability'] * 100).round(2)

    for col, default in [
        ('dk_line', ''), ('dk_odds', None), ('dk_implied_prob', None),
        ('dk_edge', None), ('ev_units', None), ('kelly_pct', None), ('dk_plus_ev', False)
    ]:
        if col not in df_results.columns:
            df_results[col] = default

    def fmt_odds(row):
        if pd.isna(row['dk_odds']) or row['dk_odds'] is None: return ''
        o = float(row['dk_odds'])
        return f"+{int(o)}" if o > 0 else str(int(o))

    df_results['DK_Odds']    = df_results.apply(fmt_odds, axis=1)
    df_results['DK_Line']    = df_results['dk_line'].fillna('')
    df_results['DK_Implied'] = df_results['dk_implied_prob'].apply(
        lambda v: f"{v:.1f}%" if pd.notna(v) and v is not None else ''
    )
    df_results['Edge_Pct']   = df_results['dk_edge'].apply(
        lambda v: round(v, 2) if pd.notna(v) and v is not None else ''
    )
    df_results['EV_Units']   = df_results['ev_units'].apply(
        lambda v: round(v, 3) if pd.notna(v) and v is not None else ''
    )
    df_results['Kelly_Pct']  = df_results['kelly_pct'].apply(
        lambda v: f"{v:.1f}%" if pd.notna(v) and v is not None and v > 0 else ''
    )
    df_results['Plus_EV']    = df_results['dk_plus_ev'].map({True: 'YES', False: ''})

    df_results = df_results.sort_values('hr_probability', ascending=False)

    if 'xgb_oof' in locals():
        prob_true, prob_pred = calibration_curve(y_train, xgb_oof, n_bins=10, strategy='quantile')
        print("[RELIABILITY (OOF)] Predicted bins:", prob_pred)
        print("                     True probabilities:", prob_true)

    print("\n" + "="*80 + f"\n🏆 TOP 10 HR PREDICTIONS FOR {TARGET_DATE} 🏆\n" + "="*80)
    for rank, (_, row) in enumerate(df_results.head(10).iterrows(), 1):
        if row.get('dk_odds') is not None and pd.notna(row['dk_odds']):
            o = row['dk_odds']
            odds_str = f"DK {row['dk_line']}: {'+'if o>0 else ''}{int(o)}"
            ev_tag   = f"  ✅ +EV +{row['dk_edge']:.1f}%" if row.get('dk_plus_ev') else f"  ❌ Edge {row['dk_edge']:.1f}%"
            dk_line  = f"  {odds_str} (Imp {row['dk_implied_prob']:.1f}%){ev_tag}"
        else:
            dk_line  = "  DK: no line"
        print(f"{rank:2d}. {row['batter_name']:<25} vs. {row['pitcher_name']:<20} @ {row['stadium']:<5} | {row['HR %']}%")
        pos_str = ", ".join([f"+{k}" for k in row['top_pos'].index])
        neg_str = ", ".join([f"-{k}" for k in row['top_neg'].index])
        print(f"    Drivers: {pos_str} | {neg_str}")
        print(f"   {dk_line}")
    print("="*80 + f"\nStats: Max={df_results['HR %'].max()}% | Mean={df_results['HR %'].mean():.2f}% | Unique={df_results['HR %'].nunique()}")

    ev_df = (
        df_results[df_results['dk_plus_ev'] == True]
        .copy()
        .sort_values('ev_units', ascending=False)
    )

    W = 82  
    print("\n" + "="*W)
    print(f"  💰 +EV HR BETS — {TARGET_DATE}   {len(ev_df)} plays found, top 10 shown, sorted by EV/u")
    print("="*W)
    if ev_df.empty:
        print("  No +EV plays — no DK lines matched or all flagged players below MIN_PA threshold.")
    else:
        print(f"  {'#':<3} {'Player (Team)':<31} {'Ln':<3} {'Odds':>6}  "
              f"{'Model':>5}  {'Mkt':>5}  {'Edge':>5}  {'EV/u':>6}  {'Kelly':>5}")
        print("  " + "─"*78)
        for rank, (_, row) in enumerate(ev_df.head(10).iterrows(), 1):
            o        = row['dk_odds']
            o_str    = (f"+{int(o)}" if o > 0 else str(int(o))) if pd.notna(o) else '?'
            ev       = row['ev_units']        if pd.notna(row.get('ev_units'))        else 0.0
            imp      = row['dk_implied_prob'] if pd.notna(row.get('dk_implied_prob')) else 0.0
            edge     = row['dk_edge']         if pd.notna(row.get('dk_edge'))         else 0.0
            kelly    = row['Kelly_Pct'] if row['Kelly_Pct'] else '—'
            player   = f"{row['batter_name']} ({row['team']})"
            print(
                f"  {rank:<3} {player:<31} {row['DK_Line']:<3} {o_str:>6}  "
                f"{row['HR %']:>4.1f}%  {imp:>4.1f}%  {edge:>+4.1f}%  "
                f"{ev:>+6.3f}  {kelly:>5}"
            )
        if len(ev_df) > 10:
            print(f"\n  ... +{len(ev_df) - 10} more in the DailyRankings sheet.")
    print("="*W)

    def _sheets_op_with_backoff(fn, max_retries=6, base_delay=10):
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as exc:
                is_quota = "429" in str(exc) or "Quota" in str(exc)
                if is_quota and attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)   
                    print(f"[SHEETS] Rate-limited (429) — retrying in {wait}s "
                          f"(attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise

    try:
        rankings_ws = sh.worksheet("DailyRankings")
        _sheets_op_with_backoff(rankings_ws.clear)
    except Exception as exc:
        if "429" in str(exc) or "Quota" in str(exc):
            raise   
        rankings_ws = sh.add_worksheet(title="DailyRankings", rows=2000, cols=20)

    upload_df = df_results.reset_index(drop=True).copy()
    upload_df.insert(0, "Rank", range(1, len(upload_df) + 1))
    upload_df = upload_df[[
        "Rank", "team", "batter_name", "pitcher_name", "stadium",
        "HR %",
        "DK_Line", "DK_Odds", "DK_Implied", "Edge_Pct",
        "EV_Units", "Kelly_Pct", "Plus_EV"
    ]]

    payload = [upload_df.columns.tolist()] + upload_df.values.tolist()
    _sheets_op_with_backoff(lambda: rankings_ws.update(payload))

    n_ev = int(df_results['dk_plus_ev'].sum())
    print(f"\n✅ Uploaded {len(upload_df)} players to DailyRankings sheet "
          f"({n_ev} marked +EV, sorted by model HR%).")

for sess in _all_sessions:
    try: sess.close()
    except: pass
print("Inference complete.")
import os
import re
import sys
import json
import time
import zlib
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import requests

APP_ID = os.environ.get("ADZUNA_APP_ID", "34768c27")
APP_KEY = os.environ.get("ADZUNA_APP_KEY", "0affde0e978f4fa15e9d21ae17746d26")
COUNTRY = "gb"
WHERE = "Scotland"
RESULTS_PER_PAGE = 50
MAX_PAGES = 100            # largest Scottish category (trade-construction) needs about 86 pages
SLEEP_SECONDS = 3          # keeps under Adzuna's ~25 requests per minute
MAX_RETRIES = 5
USER_AGENT = "UWS-MSc-research-skillgap (contact: B01842867@studentmail.uws.ac.uk)"

# MinHash near-duplicate settings (conservative, as in the specification)
NUM_PERM = 128
BANDS = 32                 # 32 bands x 4 rows -> pairs above ~0.42 Jaccard become candidates
SHINGLE = 5                # word 5-grams
NEAR_DUP_THRESHOLD = 0.85  # confirmed with exact Jaccard on the candidates
STALE_DAYS = 90            # adverts older than this are flagged, not removed

BASE = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
DB_PATH = DATA_DIR / "jobs.duckdb"
LOG_PATH = DATA_DIR / "run_log.csv"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+44\s?\(?0?\)?\s?|\b0)(\d[\s-]?){8,9}\d\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
# Full or outward-only UK postcode, upper case only, so words like "a1" or "g2" in prose are not matched
POSTCODE_RE = re.compile(r"\b((?:AB|DD|DG|EH|FK|G|HS|IV|KA|KW|KY|ML|PA|PH|TD|ZE)\d{1,2}[A-Z]?)(?:\s*(\d[A-Z]{2}))?\b")

AGENCY_WORDS = ("recruit", "staffing", "personnel", "resourcing", "talent", "search", "appointments", "employment")
COMPANY_SUFFIX_RE = re.compile(r"\b(ltd|limited|plc|llp|llc|inc|group|uk|scotland|the)\b")


# ---------------------------------------------------------------- extract

def check_credentials():
    if not APP_ID or not APP_KEY:
        sys.exit("Set ADZUNA_APP_ID and ADZUNA_APP_KEY as environment variables before running.")


def get(url, params):
    full = {"app_id": APP_ID, "app_key": APP_KEY, "content-type": "application/json"}
    full.update(params)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    wait = SLEEP_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=full, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"  HTTP {r.status_code}, retry {attempt}/{MAX_RETRIES} in {wait}s", flush=True)
                time.sleep(wait)
                wait *= 2
                continue
            print(f"  HTTP {r.status_code} for {url}: {r.text[:200]}", flush=True)
            return None
        except requests.RequestException as e:
            print(f"  Request error ({attempt}/{MAX_RETRIES}): {e}", flush=True)
            time.sleep(wait)
            wait *= 2
    print("  Giving up on this request (daily quota may be used up).", flush=True)
    return None


def fetch_categories():
    data = get(f"{BASE}/categories", {})
    if not data:
        return [None]
    tags = [c["tag"] for c in data.get("results", []) if c.get("tag")]
    return tags or [None]


def extract(snapshot):
    """Download pages to data/raw/<snapshot>/. Skips pages already on disk, so an
    interrupted run can simply be started again the same day and it resumes."""
    snap_dir = RAW_DIR / snapshot
    snap_dir.mkdir(parents=True, exist_ok=True)
    for cat in fetch_categories():
        label = cat or "all"
        for page in range(1, MAX_PAGES + 1):
            f = snap_dir / f"{label}_p{page:03d}.json"
            if f.exists():
                data = json.loads(f.read_text(encoding="utf-8"))
            else:
                params = {"where": WHERE, "results_per_page": RESULTS_PER_PAGE, "sort_by": "date"}
                if cat:
                    params["category"] = cat
                data = get(f"{BASE}/search/{page}", params)
                time.sleep(SLEEP_SECONDS)
                if not data:
                    break
                if data.get("results"):
                    f.write_text(json.dumps(data), encoding="utf-8")
            results = data.get("results", [])
            total = data.get("count", 0)
            if not results:
                break
            print(f"{snapshot} | {label} | page {page} | {len(results)} ads | reported total {total}", flush=True)
            if page * RESULTS_PER_PAGE >= (total or 0):
                break


def read_raw(snapshot):
    """Read every saved page for a snapshot. Returns records and a per-category log."""
    snap_dir = RAW_DIR / snapshot
    files = sorted(snap_dir.glob("*_p*.json"))
    if not files:
        sys.exit(f"No raw files in {snap_dir}. Check the folder name and that you run from the project folder.")
    records, log = [], {}
    for f in files:
        label, page = f.stem.rsplit("_p", 1)
        data = json.loads(f.read_text(encoding="utf-8"))
        results = data.get("results", [])
        for r in results:
            r["_category_query"] = label
            r["_page"] = int(page)
        records.extend(results)
        entry = log.setdefault(label, {"snapshot": snapshot, "category": label, "reported_count": 0, "collected": 0})
        entry["reported_count"] = data.get("count", entry["reported_count"])
        entry["collected"] += len(results)
    log_rows = list(log.values())
    for row in log_rows:
        row["complete"] = row["collected"] >= (row["reported_count"] or 0) - RESULTS_PER_PAGE
    return records, log_rows


# ---------------------------------------------------------------- transform

def scrub(text):
    if not isinstance(text, str):
        return ""
    text = HTML_RE.sub(" ", text)
    text = EMAIL_RE.sub("[email removed]", text)
    text = PHONE_RE.sub("[phone removed]", text)
    text = URL_RE.sub("[url removed]", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_title(title):
    t = scrub(title).lower()
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"[^a-z0-9+#/& -]", " ", t)
    return SPACE_RE.sub(" ", t).strip()


def employer_key(name):
    """Fingerprint for employer names: lower case, no punctuation or legal suffixes, sorted tokens."""
    if not isinstance(name, str) or not name.strip():
        return None
    t = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    t = COMPANY_SUFFIX_RE.sub(" ", t)
    tokens = sorted(set(t.split()))
    return " ".join(tokens) or None


def find_postcode(display_name):
    if isinstance(display_name, str):
        m = POSTCODE_RE.search(display_name)
        if m:
            return " ".join(g for g in m.groups() if g)
    return None


def transform(records, snapshot):
    snap_date = pd.Timestamp(snapshot, tz="UTC")
    rows = []
    for r in records:
        loc = r.get("location") or {}
        area = loc.get("area") or []
        company = scrub((r.get("company") or {}).get("display_name"))
        category = r.get("category") or {}
        desc = scrub(r.get("description"))
        title_clean = clean_title(r.get("title"))
        rows.append({
            "job_id": str(r.get("id")),
            "source": "adzuna",
            "title_raw": r.get("title"),
            "title_clean": title_clean,
            "description": desc,
            "employer": company or None,
            "employer_key": employer_key(company),
            "is_agency": any(w in company.lower() for w in AGENCY_WORDS) if company else False,
            "location_raw": loc.get("display_name"),
            "area_country": area[0] if len(area) > 0 else None,
            "area_region": area[1] if len(area) > 1 else None,
            "local_authority": area[2] if len(area) > 2 else None,
            "area_town": area[3] if len(area) > 3 else None,
            "area_district": area[4] if len(area) > 4 else None,
            "geo_level": len(area),
            "postcode": find_postcode(loc.get("display_name")),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "category_tag": category.get("tag"),
            "category_label": category.get("label"),
            "category_query": r.get("_category_query"),
            "contract_type": r.get("contract_type"),
            "contract_time": r.get("contract_time"),
            "salary_min": r.get("salary_min"),
            "salary_max": r.get("salary_max"),
            "salary_is_predicted": str(r.get("salary_is_predicted")) == "1",
            "date_posted": r.get("created"),
            "redirect_url": r.get("redirect_url"),
            "snapshot": snapshot,
            "text_hash": hashlib.sha1(f"{title_clean}|{desc.lower()}|{loc.get('display_name')}".encode("utf-8")).hexdigest(),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce", utc=True)
    df["age_days"] = (snap_date - df["date_posted"]).dt.days
    df["is_stale"] = df["age_days"] > STALE_DAYS
    before = df.copy()  # everything downloaded, for before/after quality figures

    df = df[df["area_region"].fillna("").str.contains("Scotland", case=False) |
            df["location_raw"].fillna("").str.contains("Scotland", case=False) |
            df["postcode"].notna()]
    df = df.drop_duplicates(subset="job_id", keep="first")
    df = df.drop_duplicates(subset="text_hash", keep="first")
    df = flag_near_duplicates(df.reset_index(drop=True))
    return before, df


# ---------------------------------------------------------------- MinHash near-duplicates

_P = (1 << 61) - 1
_rng = np.random.default_rng(42)
_A = _rng.integers(1, 1 << 32, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, 1 << 32, NUM_PERM, dtype=np.uint64)


def shingles(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < SHINGLE:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def minhash(sh):
    if not sh:
        return np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    x = np.array([zlib.crc32(s.encode()) for s in sh], dtype=np.uint64)
    h = (np.outer(_A, x) + _B[:, None]) % np.uint64(4294967311)
    return h.min(axis=1)


def flag_near_duplicates(df):
    """MinHash + LSH over title + description, within the same local authority. Each row in a cluster keeps
    near_dup_group (the job_id of the earliest-posted member); is_near_dup is True for
    every member except that one. Rows are flagged, not deleted, so the threshold can be tuned."""
    docs = (df["title_clean"].fillna("") + " " + df["description"].fillna("")).tolist()
    sets = [shingles(d) for d in docs]
    sigs = np.vstack([minhash(s) for s in sets])
    # Cedefop rule: a duplicate must match on description AND location
    place = (df["local_authority"].fillna(df["location_raw"].fillna(""))).tolist()
    rows = NUM_PERM // BANDS
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    checked = set()
    for b in range(BANDS):
        buckets = {}
        band = sigs[:, b * rows:(b + 1) * rows]
        for i, key in enumerate(map(bytes, band)):
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 500:
                continue
            for i, j in ((members[x], members[y]) for x in range(len(members)) for y in range(x + 1, len(members))):
                pair = (i, j)
                if pair in checked:
                    continue
                checked.add(pair)
                a, c = sets[i], sets[j]
                if place[i] != place[j]:
                    continue
                if a and c and len(a & c) / len(a | c) >= NEAR_DUP_THRESHOLD:
                    parent[find(j)] = find(i)

    roots = [find(i) for i in range(len(df))]
    df = df.copy()
    df["_root"] = roots
    df = df.sort_values(["_root", "date_posted"])
    first = df.groupby("_root")["job_id"].transform("first")
    df["near_dup_group"] = first
    df["is_near_dup"] = df["job_id"] != first
    return df.drop(columns="_root").sort_index().reset_index(drop=True)


# ---------------------------------------------------------------- quality and load

def quality(df, label):
    n = len(df)
    if n == 0:
        return {"stage": label, "rows": 0}
    return {
        "stage": label,
        "rows": n,
        "duplicate_job_id_rate": round(1 - df["job_id"].nunique() / n, 4),
        "exact_duplicate_text_rate": round(1 - df["text_hash"].nunique() / n, 4),
        "near_duplicate_rate": round(df["is_near_dup"].mean(), 4) if "is_near_dup" in df else None,
        "missing_employer": round(df["employer"].isna().mean(), 4),
        "missing_actual_salary": round((df["salary_min"].isna() | df["salary_is_predicted"]).mean(), 4),
        "missing_contract_time": round(df["contract_time"].isna().mean(), 4),
        "missing_contract_type": round(df["contract_type"].isna().mean(), 4),
        "missing_local_authority": round(df["local_authority"].isna().mean(), 4),
        "missing_postcode": round(df["postcode"].isna().mean(), 4),
        "missing_lat_lon": round(df["latitude"].isna().mean(), 4),
        "stale_share": round(df["is_stale"].mean(), 4),
        "outside_scotland": int((~df["area_region"].fillna("").str.contains("Scotland")).sum()),
    }


def load(df, snapshot):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DIR / f"snapshot_{snapshot}.csv", index=False)
    try:
        df.to_parquet(CLEAN_DIR / f"snapshot_{snapshot}.parquet", index=False)
    except ImportError:
        print("pyarrow not installed: skipped Parquet (run: pip install pyarrow)")

    con = duckdb.connect(str(DB_PATH))
    con.register("incoming", df)
    cols = con.execute("SELECT * FROM incoming LIMIT 0").description
    expected = [c[0] for c in cols] + ["first_seen", "last_seen"]
    exists = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'postings'").fetchone()[0]
    if exists:
        have = [r[0] for r in con.execute("DESCRIBE postings").fetchall()]
        if have != expected:
            print("Database schema changed since last run: rebuilding the postings table.")
            con.execute("DROP TABLE postings")
            exists = 0
    if not exists:
        con.execute("CREATE TABLE postings AS SELECT *, snapshot AS first_seen, snapshot AS last_seen FROM incoming WHERE 1=0")

    con.execute("""
        UPDATE postings SET last_seen = GREATEST(postings.last_seen, i.snapshot)
        FROM incoming i WHERE postings.job_id = i.job_id
    """)
    con.execute("""
        INSERT INTO postings
        SELECT i.*, i.snapshot, i.snapshot FROM incoming i
        WHERE i.job_id NOT IN (SELECT job_id FROM postings)
    """)
    total = con.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
    out = (CLEAN_DIR / "postings_all").as_posix()
    con.execute(f"COPY postings TO '{out}.csv' (HEADER, DELIMITER ',')")
    con.execute(f"COPY postings TO '{out}.parquet' (FORMAT PARQUET)")
    con.close()
    return total


DICTIONARY = {
    "job_id": "Adzuna advert id (unique key)",
    "source": "Data source (adzuna)",
    "title_raw": "Job title as published",
    "title_clean": "Lower-case title without bracketed text or symbols",
    "description": "Advert snippet (Adzuna truncates to ~500 chars); HTML, emails, phones and URLs removed",
    "employer": "Employer display name",
    "employer_key": "Normalised employer fingerprint for matching",
    "is_agency": "True if employer name looks like a recruitment agency",
    "location_raw": "Location as published",
    "area_country": "Adzuna geography level 1",
    "area_region": "Adzuna geography level 2 (Scotland)",
    "local_authority": "Adzuna geography level 3 (mostly council area); map to SDS College Region / ITL3 downstream",
    "area_town": "Adzuna geography level 4",
    "area_district": "Adzuna geography level 5",
    "geo_level": "Number of geography levels given (1 = country only)",
    "postcode": "Scottish postcode found in the location text, if any",
    "latitude": "Latitude", "longitude": "Longitude",
    "category_tag": "Adzuna category tag", "category_label": "Adzuna category label",
    "category_query": "Category used in the API query that returned the advert",
    "contract_type": "permanent / contract", "contract_time": "full_time / part_time",
    "salary_min": "Minimum salary (GBP per year)", "salary_max": "Maximum salary (GBP per year)",
    "salary_is_predicted": "True if Adzuna estimated the salary rather than the advert stating it",
    "date_posted": "Advert creation date (UTC)",
    "age_days": "Days between date_posted and snapshot",
    "is_stale": f"True if older than {STALE_DAYS} days at snapshot",
    "redirect_url": "Link to original advert",
    "snapshot": "Collection date (YYYY-MM-DD)",
    "text_hash": "SHA-1 of title + description + location (exact-duplicate key)",
    "near_dup_group": "job_id of the earliest advert in its MinHash near-duplicate cluster",
    "is_near_dup": f"True if Jaccard >= {NEAR_DUP_THRESHOLD} with an earlier advert (exclude for counts)",
    "first_seen": "First snapshot containing the advert (postings_all only)",
    "last_seen": "Latest snapshot containing the advert (postings_all only)",
}


def process(snapshot):
    records, log_rows = read_raw(snapshot)
    print(f"\n[{snapshot}] read {len(records)} raw records from {len(log_rows)} categories")
    incomplete = [r["category"] for r in log_rows if not r["complete"]]
    if incomplete:
        print(f"  WARNING incomplete categories (download again to finish): {', '.join(incomplete)}")
    before, df = transform(records, snapshot)
    if df.empty:
        print("Nothing to load.")
        return
    q = pd.DataFrame([quality(before, "raw"), quality(df, "clean"),
                      quality(df[~df["is_near_dup"]], "clean_excluding_near_dups")])
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    q.to_csv(CLEAN_DIR / f"quality_{snapshot}.csv", index=False)
    total = load(df, snapshot)
    pd.DataFrame(DICTIONARY.items(), columns=["field", "description"]).to_csv(CLEAN_DIR / "data_dictionary.csv", index=False)

    log = pd.DataFrame(log_rows)
    log["raw_records"] = len(records)
    log["clean_records"] = len(df)
    log["near_duplicates"] = int(df["is_near_dup"].sum())
    log["db_total"] = total
    log["run_time_utc"] = datetime.now(timezone.utc).isoformat()
    log.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

    print(f"  clean rows {len(df)} | near-duplicates flagged {int(df['is_near_dup'].sum())} | "
          f"stale (> {STALE_DAYS} days) {int(df['is_stale'].sum())} | database total {total}")
    print(q.T.to_string(header=False))
    print(f"  wrote {CLEAN_DIR / f'snapshot_{snapshot}.csv'} and {CLEAN_DIR / 'postings_all.csv'}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    args = sys.argv[1:]
    if args and args[0] == "--all-raw":
        snaps = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    elif args:
        snaps = [args[0]]
    else:
        check_credentials()
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        extract(snapshot)
        snaps = [snapshot]
    for s in snaps:
        process(s)


if __name__ == "__main__":
    main()

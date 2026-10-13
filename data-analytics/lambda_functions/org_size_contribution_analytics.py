"""Size & Contribution Analytics API for Organization Dashboard (Issue #376).

Returns data for the "Size & Contribution" tab:
1. organizations_by_size: Organization counts by org_size category.
2. collaborator_vs_contributor: Independent counts and percentages of collaborator
   and contributor organizations.

Response Shape Contract:
- No Custom date range provided (e.g., {}):
  Returns fixed buckets ("7D", "30D", "1Y", "All") plus an empty "Custom" bucket.
- Any Custom date range provided (size_*, contribution_*, or both):
  Returns only the "Custom" key. Fixed buckets are omitted.
  - If size_start_date/size_end_date is provided -> organizations_by_size populated.
  - If contribution_start_date/contribution_end_date is provided -> collaborator_vs_contributor populated.
  - If both are provided -> both charts are populated independently from their respective ranges.

Filters:
- country: Country name or code (or "ALL"). Joins organizations -> states -> countries.
- organization_type: "non_profit", "for_profit", or "ALL".
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta

import pandas as pd

try:
    import psycopg2
except ImportError:
    psycopg2 = None

logger = logging.getLogger(__name__)

# Directory resolution for mock CSV data
DEFAULT_MOCK_DATA_DIR = os.environ.get(
    "MOCK_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sql")),
)

FIXED_BUCKETS = ["7D", "30D", "1Y", "All"]
ORG_COLUMNS = [
    "org_id",
    "org_name",
    "state_id",
    "org_type",
    "org_size",
    "is_collaborator",
    "created_at",
]

US_ALIASES = {
    "USA",
    "US",
    "UNITED_STATES",
    "UNITED_STATES_OF_AMERICA",
}


# ---------------------------------------------------------------------------
# Helpers & Data Normalization
# ---------------------------------------------------------------------------

def _to_bool(value):
    """Convert various boolean representations (string, int, bool) to Python bool."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    s = str(value).strip().upper()
    return s in ("TRUE", "1", "T", "YES", "Y")


def _normalize_org_type(value):
    """Normalize org_type string to 'non_profit', 'for_profit', etc."""
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    if s == "nonprofit":
        return "non_profit"
    if s == "forprofit":
        return "for_profit"
    return s


def _normalize_string(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip().upper().replace(" ", "_").replace("-", "_")


def _find_csv(dir_path, primary_name, fallback_name):
    """Find CSV by checking primary name, then fallback name."""
    p1 = os.path.join(dir_path, primary_name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(dir_path, fallback_name)
    if os.path.exists(p2):
        return p2
    return p1


# ---------------------------------------------------------------------------
# Data Loading (Mock CSV & PostgreSQL)
# ---------------------------------------------------------------------------

def load_mock_data(sql_dir=None):
    """Load mock CSV data from sql_dir or MOCK_DATA_DIR environment variable."""
    dir_path = sql_dir or os.environ.get("MOCK_DATA_DIR", DEFAULT_MOCK_DATA_DIR)

    org_path = _find_csv(dir_path, "organizations.csv", "organization.csv")
    state_path = _find_csv(dir_path, "states.csv", "state.csv")
    country_path = _find_csv(dir_path, "countries.csv", "country.csv")

    try:
        org_df = pd.read_csv(org_path, dtype={"state_id": str})
    except (pd.errors.EmptyDataError, FileNotFoundError):
        org_df = pd.DataFrame(columns=ORG_COLUMNS + ["is_contributor"])

    try:
        state_df = pd.read_csv(state_path, dtype={"state_id": str, "country_id": str})
    except (pd.errors.EmptyDataError, FileNotFoundError):
        state_df = pd.DataFrame(columns=["state_id", "country_id", "state_code"])

    try:
        country_df = pd.read_csv(country_path, dtype={"country_id": str})
    except (pd.errors.EmptyDataError, FileNotFoundError):
        country_df = pd.DataFrame(columns=["country_id", "country_code", "country_name"])

    # Clean organizations
    if not org_df.empty and "created_at" in org_df.columns:
        org_df["created_at"] = pd.to_datetime(
            org_df["created_at"], errors="coerce", utc=True, format="mixed"
        ).dt.tz_localize(None)
        org_df = org_df.dropna(subset=["created_at"])
    else:
        org_df["created_at"] = pd.Series(dtype="datetime64[ns]")

    if "is_collaborator" in org_df.columns:
        org_df["is_collaborator"] = org_df["is_collaborator"].apply(_to_bool)
    else:
        org_df["is_collaborator"] = False

    # Graceful degradation if is_contributor is missing from dataset
    if "is_contributor" in org_df.columns:
        org_df["is_contributor"] = org_df["is_contributor"].apply(_to_bool)
    else:
        org_df["is_contributor"] = False

    if "org_type" in org_df.columns:
        org_df["org_type"] = org_df["org_type"].apply(_normalize_org_type)
    else:
        org_df["org_type"] = ""

    if "org_size" in org_df.columns:
        org_df["org_size"] = org_df["org_size"].fillna("").astype(str).str.strip().str.lower()
    else:
        org_df["org_size"] = ""

    # Ensure join columns exist
    if "country_code" not in country_df.columns:
        country_df["country_code"] = ""
    if "country_name" not in country_df.columns:
        country_df["country_name"] = ""
    if "state_code" not in state_df.columns:
        state_df["state_code"] = ""

    state_clean = state_df[["state_id", "country_id", "state_code"]].drop_duplicates("state_id")
    country_clean = country_df[["country_id", "country_code", "country_name"]].drop_duplicates("country_id")

    merged = org_df.merge(state_clean, on="state_id", how="left")
    merged = merged.merge(country_clean, on="country_id", how="left")
    return merged


def load_db_data():
    """Load real data from PostgreSQL using psycopg2."""
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed; install it or set USE_MOCK_DATA=true")

    schema = os.environ.get("ORG_ANALYTICS_SCHEMA", "virginia_dev_saayam_rdbms")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError(f"Invalid ORG_ANALYTICS_SCHEMA: {schema}")

    query = f"""
        SELECT o.org_id, o.org_name, o.state_id, o.org_type, o.org_size,
               o.is_collaborator,
               COALESCE(o.is_contributor, false) AS is_contributor,
               o.created_at,
               s.state_code,
               c.country_code, c.country_name
        FROM {schema}.organizations o
        LEFT JOIN state s ON o.state_id = s.state_id
        LEFT JOIN country c ON s.country_id = c.country_id
    """

    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ.get("DB_NAME", "saayam"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty and "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True, format="mixed"
        ).dt.tz_localize(None)
        df = df.dropna(subset=["created_at"])
    df["is_collaborator"] = df["is_collaborator"].apply(_to_bool)
    df["is_contributor"] = df["is_contributor"].apply(_to_bool)
    df["org_type"] = df["org_type"].apply(_normalize_org_type)
    df["org_size"] = df["org_size"].fillna("").astype(str).str.strip().str.lower()
    return df


def load_data(sql_dir=None):
    use_mock = os.environ.get("USE_MOCK_DATA", "true").strip().lower() == "true"
    if use_mock:
        return load_mock_data(sql_dir)
    return load_db_data()


# ---------------------------------------------------------------------------
# Filtering Logic
# ---------------------------------------------------------------------------

def apply_country_filter(df, country):
    """Filter by country code or country name; None, '', or 'ALL' disables filter."""
    if country is None:
        return df
    if not isinstance(country, str):
        raise ValueError("country must be a string")
    s = country.strip()
    if s == "" or s.upper() == "ALL":
        return df

    target = _normalize_string(s)
    codes = df["country_code"].fillna("").map(_normalize_string)
    names = df["country_name"].fillna("").map(_normalize_string)
    match_mask = (codes == target) | (names == target)

    # If filtering for USA, also match states with US- prefix in state_code
    if target in US_ALIASES:
        if "state_code" in df.columns:
            is_us_state = df["state_code"].fillna("").str.upper().str.startswith("US-")
            match_mask = match_mask | is_us_state

    return df[match_mask]


def apply_org_type_filter(df, organization_type):
    """Filter by organization_type (non_profit / for_profit / ALL)."""
    if organization_type is None:
        return df
    if not isinstance(organization_type, str):
        raise ValueError("organization_type must be a string")
    s = organization_type.strip()
    if s == "" or s.upper() == "ALL":
        return df

    target = _normalize_org_type(s)
    return df[df["org_type"] == target]


def apply_filters(df, country=None, organization_type=None):
    df = apply_country_filter(df, country)
    df = apply_org_type_filter(df, organization_type)
    return df


# ---------------------------------------------------------------------------
# Time Windows & Aggregation
# ---------------------------------------------------------------------------

def get_bucket_range(bucket, now=None):
    """Return (start_datetime, end_datetime) for a fixed bucket."""
    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if bucket == "7D":
        return today - timedelta(days=7), now
    if bucket == "30D":
        return today - timedelta(days=30), now
    if bucket == "1Y":
        try:
            one_year_ago = today.replace(year=today.year - 1)
        except ValueError:
            one_year_ago = today.replace(year=today.year - 1, day=28)
        return one_year_ago, now
    if bucket == "All":
        return None, now
    raise ValueError(f"Unknown bucket: {bucket}")


def filter_by_window(df, start_dt, end_dt):
    """Filter dataframe rows where created_at is within [start_dt, end_dt]."""
    if df.empty:
        return df
    windowed = df[df["created_at"] <= end_dt]
    if start_dt is not None:
        windowed = windowed[windowed["created_at"] >= start_dt]
    return windowed


def build_organizations_by_size(df):
    """One row per org_size category present in that window's data.

    Returns empty list if no organizations in window.
    """
    if df.empty or len(df) == 0:
        return []

    valid_sizes = df[df["org_size"] != ""]
    if valid_sizes.empty:
        return []

    grouped = valid_sizes.groupby("org_size").size().reset_index(name="count")
    # Order by count descending, then size alphabetically for stability
    grouped = grouped.sort_values(by=["count", "org_size"], ascending=[False, True])

    return [
        {"size": str(row["org_size"]), "count": int(row["count"])}
        for _, row in grouped.iterrows()
    ]


def build_collaborator_vs_contributor(df):
    """Compute independent counts for Collaborator and Contributor.

    Percentages calculated against total organization count in window.
    Returns empty list if total == 0.
    """
    total = len(df)
    if total == 0:
        return []

    collab_count = int((df["is_collaborator"] == True).sum())
    contrib_count = int((df["is_contributor"] == True).sum())

    collab_pct = round((collab_count / total) * 100, 1)
    contrib_pct = round((contrib_count / total) * 100, 1)

    return [
        {"type": "Collaborator", "count": collab_count, "percentage": collab_pct},
        {"type": "Contributor", "count": contrib_count, "percentage": contrib_pct},
    ]


def build_bucket(df, window_start, window_end):
    """Build response object for a single window."""
    windowed = filter_by_window(df, window_start, window_end)
    return {
        "organizations_by_size": build_organizations_by_size(windowed),
        "collaborator_vs_contributor": build_collaborator_vs_contributor(windowed),
    }


# ---------------------------------------------------------------------------
# Validation & Parsing
# ---------------------------------------------------------------------------

def parse_date_or_error(date_str, field_name):
    """Parse YYYY-MM-DD string or return error message."""
    if not isinstance(date_str, str):
        return None, f"Invalid date format for {field_name}. Use YYYY-MM-DD"
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return dt, None
    except ValueError:
        return None, f"Invalid date format for {field_name}. Use YYYY-MM-DD"


def validate_date_range(start_str, end_str, label):
    """Validate a custom date range pair.

    Returns (start_dt, end_dt, error_message).
    If both are None, returns (None, None, None).
    """
    if start_str is None and end_str is None:
        return None, None, None

    if start_str is None:
        return None, None, f"{label}_start_date is required when {label}_end_date is provided"
    if end_str is None:
        return None, None, f"{label}_end_date is required when {label}_start_date is provided"

    start_dt, err = parse_date_or_error(start_str, f"{label}_start_date")
    if err:
        return None, None, err

    end_dt, err = parse_date_or_error(end_str, f"{label}_end_date")
    if err:
        return None, None, err

    if start_dt > end_dt:
        return None, None, f"{label}_start_date must be before or equal to {label}_end_date"

    # End of day boundary (23:59:59.999999)
    end_of_day = end_dt + timedelta(days=1) - timedelta(microseconds=1)
    return start_dt, end_of_day, None


def parse_event(event):
    """Extract params from raw dict, API Gateway proxy event, or query parameters."""
    if event is None:
        return {}, None
    if not isinstance(event, dict):
        return None, "Request must be a JSON object"

    # Direct dict call without API Gateway wrapper
    if not any(k in event for k in ("body", "queryStringParameters", "httpMethod")):
        return event, None

    params = dict(event.get("queryStringParameters") or {})
    body = event.get("body")
    if isinstance(body, str):
        if body.strip():
            try:
                parsed_body = json.loads(body)
                if isinstance(parsed_body, dict):
                    params.update(parsed_body)
                else:
                    return None, "Request body must be a JSON object"
            except json.JSONDecodeError:
                return None, "Request body must be valid JSON"
    elif isinstance(body, dict):
        params.update(body)

    return params, None


def build_response(status_code, body):
    """Construct standard AWS Lambda proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body) if isinstance(body, (dict, list)) else body,
    }


# ---------------------------------------------------------------------------
# Lambda Handler
# ---------------------------------------------------------------------------

def handler(event, context=None, now=None):
    """AWS Lambda entry point."""
    try:
        params, err = parse_event(event)
        if err:
            return build_response(400, {"error": err})

        country = params.get("country")
        organization_type = params.get("organization_type")

        # Validate custom date ranges independently
        size_start, size_end, err = validate_date_range(
            params.get("size_start_date"), params.get("size_end_date"), "size"
        )
        if err:
            return build_response(400, {"error": err})

        contrib_start, contrib_end, err = validate_date_range(
            params.get("contribution_start_date"),
            params.get("contribution_end_date"),
            "contribution",
        )
        if err:
            return build_response(400, {"error": err})

        # Load dataset
        try:
            df = load_data()
        except Exception as exc:
            logger.exception("Error loading data: %s", exc)
            return build_response(500, {"error": f"Failed to load data: {str(exc)}"})

        # Apply common filters (country, organization_type)
        try:
            filtered_df = apply_filters(df, country=country, organization_type=organization_type)
        except ValueError as val_err:
            return build_response(400, {"error": str(val_err)})

        has_size_custom = size_start is not None and size_end is not None
        has_contrib_custom = contrib_start is not None and contrib_end is not None
        is_custom = has_size_custom or has_contrib_custom

        # Response Shape 1: Custom range supplied -> Return ONLY "Custom" key
        if is_custom:
            size_data = []
            contrib_data = []

            if has_size_custom:
                size_window_df = filter_by_window(filtered_df, size_start, size_end)
                size_data = build_organizations_by_size(size_window_df)

            if has_contrib_custom:
                contrib_window_df = filter_by_window(filtered_df, contrib_start, contrib_end)
                contrib_data = build_collaborator_vs_contributor(contrib_window_df)

            response = {
                "Custom": {
                    "organizations_by_size": size_data,
                    "collaborator_vs_contributor": contrib_data,
                }
            }
            return build_response(200, response)

        # Response Shape 2: No custom params -> Return 5 top-level keys
        response = {}
        for bucket in FIXED_BUCKETS:
            w_start, w_end = get_bucket_range(bucket, now=now)
            response[bucket] = build_bucket(filtered_df, w_start, w_end)

        response["Custom"] = {
            "organizations_by_size": [],
            "collaborator_vs_contributor": [],
        }

        return build_response(200, response)

    except Exception as exc:
        logger.exception("Unhandled error in Size & Contribution handler: %s", exc)
        return build_response(500, {"error": f"Internal server error: {str(exc)}"})


# Alias for AWS Lambda standard configuration
lambda_handler = handler


# ---------------------------------------------------------------------------
# Local Testing Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["USE_MOCK_DATA"] = "true"
    print("=" * 70)
    print("Running Size & Contribution Analytics API Sample Tests")
    print("=" * 70)

    test_events = [
        ("1. Empty body (All fixed buckets + empty Custom)", {}),
        ("2. Country filter (USA)", {"country": "USA"}),
        ("3. Org-type filter (non_profit)", {"organization_type": "non_profit"}),
        (
            "4. Size custom range only (Custom-only response)",
            {"size_start_date": "2024-01-01", "size_end_date": "2025-12-31"},
        ),
        (
            "5. Contribution custom range only (Custom-only response)",
            {"contribution_start_date": "2025-01-01", "contribution_end_date": "2025-12-31"},
        ),
        (
            "6. Both custom ranges simultaneously",
            {
                "country": "USA",
                "organization_type": "non_profit",
                "size_start_date": "2024-01-01",
                "size_end_date": "2025-12-31",
                "contribution_start_date": "2025-01-01",
                "contribution_end_date": "2025-12-31",
            },
        ),
        (
            "7. Invalid date validation (start > end)",
            {"size_start_date": "2026-05-01", "size_end_date": "2026-01-01"},
        ),
        (
            "8. Missing half of date pair",
            {"size_start_date": "2026-01-01"},
        ),
    ]

    for label, evt in test_events:
        print(f"\n--- {label} ---")
        res = handler(evt)
        print(f"Status Code: {res['statusCode']}")
        parsed = json.loads(res["body"]) if isinstance(res["body"], str) else res["body"]
        print(json.dumps(parsed, indent=2))

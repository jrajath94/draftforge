"""SEC EDGAR finance loader (XBRL company-facts API).

Primary fallback for finance-domain training data when FinOpsGym / HF
finance datasets are unavailable or unauthenticated. SEC EDGAR is fully
public; no API key, no rate-limit fee. We honor their fair-access policy:

  - User-Agent header required (any contact string).
  - Cap to ~6 req/sec; this loader sleeps 0.15s between CIKs.

Strategy:
  1. GET /api/xbrl/companyfacts/CIK{cik}.json for a list of issuers
  2. Pull USD-denominated us-gaap concepts (Revenues, NetIncomeLoss, ...)
  3. Filter to 10-K filings (form=10-K, fp=FY)
  4. Emit one Q&A per (entity, concept, fiscal-year) — domain=finance, source=edgar

Modes:
  - Network (default): ciks → live fetch → return examples
  - Offline: path=... → JSONL cache replay
  - Hybrid: live fetch + write JSONL cache, then `path=cache` for repeat runs

Usage:
    from data.sources.edgar import load_edgar_finance, write_edgar_cache
    examples = load_edgar_finance()  # default ~8 issuers x 5 concepts x 12yrs
    write_edgar_cache(examples, Path("data/finance/edgar_cache.jsonl"))
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from data.types import Example

EDGAR_BASE = "https://data.sec.gov"
EDGAR_TIMEOUT_SEC = 30.0
DEFAULT_USER_AGENT = "DraftForge/0.1 (research; contact@example.com)"

# Default issuer sweep. 8 large-cap companies with broad 10-K history
# and clean us-gaap coverage.
DEFAULT_CIKS: list[str] = [
    "0000320193",  # Apple
    "0000789019",  # Microsoft
    "0001018724",  # Amazon
    "0001045810",  # NVIDIA
    "0000051143",  # IBM
    "0000034088",  # 3M
    "0000093410",  # Wells Fargo
    "0000078003",  # Pfizer
]

# XBRL concepts → human-readable label. Limited to metrics broadly
# reported across industries; extend cautiously.
DEFAULT_CONCEPTS: list[tuple[str, str]] = [
    ("Revenues", "revenues"),
    ("NetIncomeLoss", "net income"),
    ("Assets", "total assets"),
    ("StockholdersEquity", "stockholders' equity"),
    ("CashAndCashEquivalentsAtCarryingValue", "cash and equivalents"),
]


def _http_get_json(url: str, user_agent: str, timeout: float = EDGAR_TIMEOUT_SEC) -> dict:
    """GET JSON from URL with SEC-compliant User-Agent header."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _format_usd(val: int) -> str:
    """Render a USD integer as billions / millions / raw for readability."""
    if abs(val) >= 1_000_000_000:
        return f"${val / 1_000_000_000:.2f} billion"
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.2f} million"
    return f"${val:,}"


def _facts_to_qa(
    entity_name: str,
    cik: str,
    concept_data: dict,
    concept_tag: str,
    concept_label: str,
    max_per_concept: int = 12,
) -> list[Example]:
    """Generate one Q&A per (entity, concept, fiscal-year) from XBRL facts."""
    out: list[Example] = []
    units = concept_data.get("units", {}).get("USD", [])
    if not units:
        return out
    annual = [f for f in units if f.get("form") == "10-K" and f.get("fp") == "FY"]
    annual.sort(key=lambda f: f.get("end", ""), reverse=True)
    for fact in annual[:max_per_concept]:
        end = fact.get("end", "")
        if not end or len(end) < 4:
            continue
        val = fact.get("val")
        if val is None:
            continue
        period_year = end[:4]
        v_str = _format_usd(int(val))
        question = (
            f"What was {entity_name}'s {concept_label} for fiscal year {period_year}? "
            f"(Source: SEC 10-K filing)"
        )
        answer = (
            f"According to {entity_name}'s 10-K filing for fiscal year {period_year}, "
            f"reported {concept_label} were {v_str} (USD)."
        )
        slug = entity_name.lower().replace(" ", "-").replace(".", "")
        out.append(
            Example(
                id=f"edgar-{slug}-{concept_tag.lower()}-{period_year}",
                domain="finance",
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                source="edgar",
                meta={
                    "cik": cik,
                    "concept": concept_tag,
                    "period_end": end,
                    "val_usd": val,
                    "form": "10-K",
                },
            )
        )
    return out


def load_edgar_finance(
    ciks: list[str] | None = None,
    path: Path | None = None,
    max_examples: int = 50_000,
    user_agent: str = DEFAULT_USER_AGENT,
    offline: bool = False,
    rate_limit_sec: float = 0.15,
) -> list[Example]:
    """Load finance Q&A from SEC EDGAR (XBRL company-facts).

    Args:
        ciks: List of SEC CIK strings (10-digit zero-padded). Defaults to
            DEFAULT_CIKS if None.
        path: Local JSONL cache file. When given (and not offline=False with
            ciks=None), reads from disk instead of fetching.
        max_examples: Cap on returned examples.
        user_agent: SEC fair-access policy requires a contact string.
        offline: Force read from `path`; fail loudly if `path` is None.
        rate_limit_sec: Sleep between CIK fetches (SEC fair-access).

    Returns:
        List of Example with domain="finance", source="edgar".
    """
    if offline or (path is not None and ciks is None):
        # Offline / cache-only path
        if path is None:
            raise ValueError("offline mode requires path (local cache location)")
        return _load_from_jsonl(path, max_examples)

    ciks_to_use = ciks if ciks is not None else DEFAULT_CIKS
    out: list[Example] = []
    blocked_count = 0
    for cik in ciks_to_use:
        if len(out) >= max_examples:
            break
        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            payload = _http_get_json(url, user_agent)
        except urllib.error.HTTPError as e:
            # SEC's WAF returns 403 for UAs with non-conforming chars
            # (notably '+' in URLs, or anything matching bot patterns).
            # Fail loudly so the operator knows their UA is the problem
            # rather than getting a silent zero return.
            if e.code in (403, 429):
                print(
                    f"[edgar] CIK {cik}: HTTP {e.code} from SEC. "
                    f"User-Agent '{user_agent}' may be blocked. "
                    f"Use the DEFAULT_USER_AGENT or a contact string without '+' chars.",
                    file=sys.stderr,
                )
                blocked_count += 1
                continue
            # Other HTTP errors: skip and continue
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            # Network down — caller can retry later, or load from cache via path=
            continue
        entity_name = payload.get("entityName", cik)
        for concept_tag, concept_label in DEFAULT_CONCEPTS:
            if len(out) >= max_examples:
                break
            concept_data = (
                payload.get("facts", {}).get("us-gaap", {}).get(concept_tag)
            )
            if not concept_data:
                continue
            qa = _facts_to_qa(
                entity_name=entity_name,
                cik=cik,
                concept_data=concept_data,
                concept_tag=concept_tag,
                concept_label=concept_label,
            )
            out.extend(qa[: max_examples - len(out)])
        time.sleep(rate_limit_sec)
    if blocked_count and not out:
        # Every CIK was blocked → don't silently return 0. Raise so the
        # caller (data pipeline) reports the real problem.
        raise RuntimeError(
            f"SEC EDGAR blocked all {blocked_count} CIK request(s) for "
            f"User-Agent '{user_agent}'. SEC WAF returns 403 for UAs "
            f"containing '+' or other non-conforming chars. Use the default "
            f"User-Agent or supply one without '+'."
        )
    return out[:max_examples]


def _load_from_jsonl(path: Path, max_examples: int) -> list[Example]:
    """Read a previously-cached EDGAR JSONL into Example list."""
    out: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if len(messages) < 2:
                continue
            out.append(
                Example(
                    id=row.get("id", f"edgar-{i:06d}"),
                    domain="finance",
                    messages=[
                        {"role": str(m["role"]), "content": str(m["content"])}
                        for m in messages
                    ],
                    source="edgar",
                    meta=row.get("meta", {}),
                )
            )
    return out


def write_edgar_cache(examples: list[Example], path: Path) -> None:
    """Persist EDGAR-derived examples to JSONL for offline replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "id": ex.id,
                        "domain": ex.domain,
                        "messages": ex.messages,
                        "source": ex.source,
                        "meta": ex.meta,
                    }
                )
                + "\n"
            )

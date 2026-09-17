"""
Small utilities shared across the autocomplete modules.

Fetching one JSON document, collapsing whitespace, reading a DOI out of free
text: none of it knows what a lipid bilayer is, and none of it decides anything
about a record. The registry-specific and record-specific logic lives in
``expsim_metadata.registries`` and ``expsim_metadata.fields``, which both build on these.
"""

import json
import random
import re
import time
import urllib.error
import urllib.request

from .constants import (
    BACKOFF_BASE,
    DEFAULT_TIMEOUT,
    DOI_RE,
    MAX_BACKOFF,
    MAX_RETRIES,
    NULLISH,
    ORCID_RE,
    RETRYABLE_STATUS,
    TRAILING,
    USER_AGENT,
)

# ---------------------------------------------------------------------------
# Talking to a registry
# ---------------------------------------------------------------------------


def _retry_delay(error, attempt):
    """Seconds to wait before the next attempt.

    Prefers a server-provided ``Retry-After`` header and otherwise falls back to
    exponential backoff with jitter, so concurrent callers do not retry in step.
    """
    headers = getattr(error, "headers", None)
    retry_after = headers.get("Retry-After") if headers is not None else None
    if retry_after:
        try:
            return min(float(retry_after), MAX_BACKOFF)
        except (TypeError, ValueError):
            pass
    return min(BACKOFF_BASE * (2**attempt), MAX_BACKOFF) + random.uniform(0, 0.5)


def fetch_json(url, timeout=DEFAULT_TIMEOUT):
    """GET a JSON document, or ``None`` when the resource is unavailable.

    Transient failures are retried; definitive ones (notably 404, meaning the
    DOI is not registered with this agency) are not.
    """
    request = urllib.request.Request(url)  # noqa: S310 - https URLs built above
    request.add_header("User-Agent", USER_AGENT)

    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                if error.code != 404:
                    print(f"  warning: {url} -> HTTP {error.code}")
                return None
            time.sleep(_retry_delay(error, attempt))
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            if attempt == MAX_RETRIES:
                print(f"  warning: {url} -> {error}")
                return None
            time.sleep(_retry_delay(error, attempt))
    return None


# ---------------------------------------------------------------------------
# Normalising what a registry or a record says
# ---------------------------------------------------------------------------


def iso_date(value: any) -> str | None:
    """Longest valid ISO prefix: ``YYYY-MM-DD``, ``YYYY-MM`` or ``YYYY``.

    Registry dates are usually complete but degrade to a year, and storing a
    year is better than storing nothing.
    """
    if value is None:
        return None
    text = str(value).strip()
    for pattern in (r"^\d{4}-\d{2}-\d{2}", r"^\d{4}-\d{2}", r"^\d{4}"):
        match = re.match(pattern, text)
        if match:
            return match.group(0)
    return None


def clean_text(value: any) -> str | None:
    """Collapse whitespace in an API-supplied string."""
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value)).strip() or None


def number(value):
    """Format a measured quantity without a spurious trailing ``.0``.

    Temperatures are written both as ``298`` and ``298.0`` across the corpus and
    the two mean the same thing; rendering them differently would give one system
    two titles.
    """
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return clean_text(value)


def normalize_doi(value):
    """Extract a bare DOI from a URL, a ``doi:`` prefix or surrounding prose."""
    match = DOI_RE.search(str(value or ""))
    return match.group(0).rstrip(TRAILING) if match else None


def normalize_orcid(value):
    match = ORCID_RE.search(str(value or ""))
    return f"https://orcid.org/{match.group(1).upper()}" if match else None


def strip_markup(value):
    """Drop markup from an abstract.

    Zenodo descriptions contain HTML and CrossRef abstracts are JATS XML;
    neither belongs in a JSON-LD description, which is plain text.
    """
    if value is None:
        return None
    text = re.sub(r"<[^>]+>", " ", str(value))
    for entity, char in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
        text = text.replace(entity, char)
    return clean_text(text)


def parse_publication(value):
    """Turn a legacy ``PUBLICATION`` field into citation entries.

    These were hand-entered and inconsistent: bare DOIs, doi.org URLs, several
    DOIs joined by semicolons, and free-text references that sometimes carry a
    DOI in parentheses and sometimes none at all. ``schema.org/citation``
    accepts Text as well as CreativeWork, so a reference with no DOI is kept
    verbatim rather than dropped.
    """
    if value is None:
        return []
    text = str(value).strip()
    if text.lower() in NULLISH:
        return []
    entries = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        dois = DOI_RE.findall(part)
        if dois:
            entries.extend(d.rstrip(TRAILING) for d in dois)
        else:
            entries.append(part)
    return entries


def dedupe(entries):
    seen, unique = set(), []
    for entry in entries:
        if entry is None:
            continue
        key = str(entry).lower()
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique

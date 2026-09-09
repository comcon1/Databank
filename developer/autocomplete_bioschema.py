"""
Script for simulation and experiment Bioschemas autocomplete.

This script fills a ``bioschema_properties`` block in a simulation or experiment
``README.yaml`` so the record can be published as a `Bioschemas Dataset
<https://bioschemas.org/profiles/Dataset/1.0-RELEASE>`_. It is the dataset-level
counterpart of ``autocomplete_metadata.py``, which does the same job for
molecules.

Values are resolved from the record's DOI:

- DataCite -- Zenodo depositions, i.e. every simulation
- CrossRef -- journal articles, i.e. most experiments
- SPDX     -- the licence list, used as the licence controlled vocabulary

and enriched from data already in the repository (composition, force field,
NMR/X-ray method, the analysis outputs present beside the README) with terms
from EDAM, CHMO and UO.

Properties that depend on where the databank is deployed -- ``identifier``,
``url``, ``@id``, ``@type``, ``@context``, ``dct:conformsTo`` and
``includedInDataCatalog`` -- are deliberately *not* written here; the web
frontend supplies them.

The script also normalises the surrounding record: dates are rewritten as quoted
ISO strings, and ``PUBLICATION`` is retired in favour of ``citation`` once its
content is safely represented there.

.. note::
   This file is meant to be used by automated workflows.

   Unlike ``autocomplete_metadata.py`` the file is **not** re-serialised. The
   generated block is appended, and only that block is rewritten in place, so
   hand-written comments elsewhere in the README survive -- most experiment
   files carry them. Runs are idempotent: a record whose content has not changed
   keeps its original ``_source.retrieved`` date.

   Upstream services answer with transient ``5xx`` errors from time to time, so
   requests are retried with exponential backoff that honours any
   ``Retry-After`` header. The retry budget can be overridden with the
   ``AUTOCOMPLETE_MAX_RETRIES`` environment variable (``0`` disables retries).
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = max(0, int(os.environ.get("AUTOCOMPLETE_MAX_RETRIES", "4")))
BACKOFF_BASE = 1.0
MAX_BACKOFF = 30.0
DEFAULT_TIMEOUT = 30
# The project contact address puts CrossRef requests in its polite pool.
USER_AGENT = (
    "FAIRMD-lipids-bioschema (+https://github.com/NMRLipids/FAIRMD_lipids; "
    "mailto:databank@nmrlipids.fi)"
)

DATACITE_URL = "https://api.datacite.org/dois/{doi}"
CROSSREF_URL = "https://api.crossref.org/works/{doi}"
SPDX_LICENSES_URL = "https://spdx.org/licenses/licenses.json"

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

EDAM_SET = "http://edamontology.org"
OBO_SET = "http://purl.obolibrary.org/obo"
QUDT_UNIT = "http://qudt.org/vocab/unit"

EDAM_SIMULATION = [
    ("topic_3892", "Biomolecular simulation"),
    ("topic_0176", "Molecular dynamics"),
    ("topic_0820", "Membrane and lipoproteins"),
    ("topic_3306", "Biophysics"),
    ("data_3870", "Trajectory data"),
]
EDAM_NMR = [
    ("topic_0593", "NMR"),
    ("topic_0820", "Membrane and lipoproteins"),
    ("topic_3306", "Biophysics"),
]
# EDAM has no small-angle-scattering term; X-ray diffraction is the closest.
EDAM_XRAY = [
    ("topic_2828", "X-ray diffraction"),
    ("topic_0820", "Membrane and lipoproteins"),
    ("topic_3306", "Biophysics"),
]

# CHMO is exact for the experimental methods and EDAM sits alongside it; neither
# covers this databank alone, since EDAM has no scattering term and CHMO no
# molecular-dynamics term. EDAM-Bioimaging was evaluated as a fallback and is
# not citable: it declares itself 1.0alpha_pre-pre-release and only 36 of its
# 505 classes carry stable numeric ids, those being EDAM terms it re-uses.
CHMO_SAXS = ("CHMO_0000204", "small-angle X-ray scattering")
CHMO_NMR = ("CHMO_0000591", "nuclear magnetic resonance spectroscopy")
CHMO_SSNMR = ("CHMO_0000614", "solid-state nuclear magnetic resonance spectroscopy")
CHMO_PDLF = ("CHMO_0001067", "proton detected separated local field spectroscopy")
CHMO_PDLF_R = ("CHMO_0001068", "R-type recoupling proton detected separated local field spectroscopy")

EDAM_MD_OP = ("operation_2476", "Molecular dynamics")
EDAM_NMR_TOPIC = ("topic_0593", "NMR")
EDAM_XRAY_TOPIC = ("topic_2828", "X-ray diffraction")

# Units. UO is preferred -- it is OBO, resolves through the same OLS endpoint as
# EDAM and CHMO, and has a square-angstrom term QUDT lacks -- but it has no
# reciprocal units, so the form factor's q axis falls back to QUDT.
#
# The density profile is a number density in nm^-3 and stays symbol-only on
# purpose: UO has nothing suitable (UO_0000177 is a solution-concentration
# grouping, UO_0000182 is mass density) and QUDT has no PER-NanoM3. QUDT's
# PER-M3 has the right dimension but is wrong by 10^27 at this scale, and a
# precise-looking wrong code is worse than an honest plain-text symbol.
UNITS = {
    "area per lipid": (f"{OBO_SET}/UO_0000324", "Å²"),
    "membrane thickness": (f"{OBO_SET}/UO_0000018", "nm"),
    "C-H bond order parameter": (f"{OBO_SET}/UO_0000186", "dimensionless"),
    "equilibration time": (f"{OBO_SET}/UO_0000150", "ns"),
    "X-ray scattering form factor": (f"{QUDT_UNIT}/PER-ANGSTROM", "Å⁻¹"),
    "total density profile": (None, "nm⁻³"),
}

# Licence of the data this repository distributes, per its own LICENSE file. An
# experiment record digitises values out of a paper: the paper carries the
# publisher's copyright, the digitised values are distributed under CC-BY-4.0.
# Those are different facts, so `license` states the dataset licence and
# `articleLicense` keeps whatever the registry said about the publication.
DATASET_LICENSE_SPDX = "CC-BY-4.0"
DATASET_LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/"

# COAR access statuses. DataCite returns these in rightsList alongside real
# licences -- and they are the most frequent entry there -- but they describe
# access, not terms of reuse.
ACCESS_RIGHTS_PREFIX = "info:eu-repo/semantics/"

# DataCite relation types pointing at the paper describing a deposition.
# IsVersionOf is excluded: it points at Zenodo's own concept DOI, which is
# another copy of this dataset rather than a citation.
CITATION_RELATIONS = frozenset({"IsSupplementTo", "IsDocumentedBy", "IsCitedBy"})

# Which analysis outputs correspond to which measured quantity.
VARIABLES = {
    "apl.json": "area per lipid",
    "thickness.json": "membrane thickness",
    "TotalDensity.json": "total density profile",
    "FormFactor.json": "X-ray scattering form factor",
    "eq_times.json": "equilibration time",
}
TRAJECTORY_FORMATS = {
    ".xtc": "application/x-xtc",
    ".trr": "application/x-trr",
    ".dcd": "application/x-dcd",
    ".nc": "application/x-netcdf",
}

# Water is in every system and says nothing about it; ions do carry meaning.
KEYWORD_SKIP = frozenset({"SOL"})

# The generated block carries no marker comment: it is found by its own
# top-level key, always the last thing in the file. LEGACY_SENTINEL is kept only
# so an older banner is stripped along with the block rather than stranded.
BLOCK_KEY = "bioschema_properties:"
BLOCK_KEY_RE = re.compile(rf"^{re.escape(BLOCK_KEY)}", re.MULTILINE)
LEGACY_SENTINEL = "# --- bioschema_properties:"

BLOCK_ORDER = [
    "name", "alternateName", "description", "sameAs", "datePublished",
    "license", "articleLicense", "publisher", "version", "creator", "citation",
    "keywords", "measurementTechnique", "variableMeasured", "distribution",
    "isBasedOn", "isPartOf", "accessRights", "_source",
]

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;,\"']+")
ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
NULLISH = frozenset({"null", "none", "na", "n/a", "??", "?", ""})
TRAILING = ".,;:)]}’\"'"


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
# SPDX licence resolution
# ---------------------------------------------------------------------------


def canonical_license_url(url):
    """Reduce a licence URL to a form comparable across sources.

    DataCite returns ``.../by/4.0``, ``.../by/4.0/`` and ``.../by/4.0/legalcode``
    for one licence while SPDX lists only the legalcode variant, so both sides
    pass through here before being compared.
    """
    text = str(url).strip().lower()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    text = re.sub(r"/(legalcode|deed)(\.[a-z-]{2,7})?$", "", text.rstrip("/"))
    return text.rstrip("/")


class SpdxIndex:
    """The SPDX licence list, indexed for lookup by URL or by name."""

    def __init__(self, payload):
        self.version = payload.get("licenseListVersion")
        self.by_url = {}
        self.by_name = {}
        self.by_id = {}
        for licence in payload.get("licenses") or []:
            self.by_id[licence["licenseId"]] = licence
            name = (licence.get("name") or "").strip().lower()
            if name:
                self.by_name.setdefault(name, licence)
            for see_also in licence.get("seeAlso") or []:
                self.by_url.setdefault(canonical_license_url(see_also), []).append(licence)

    def resolve(self, uri):
        """Return ``(licence, ambiguous)`` for a licence URI, or ``(None, False)``.

        Dozens of URLs in the SPDX list map to more than one identifier
        (GPL-2.0 vs GPL-2.0-only vs GPL-2.0-or-later, and this databank does hit
        that), so the choice is made deterministically: drop deprecated ids,
        drop legacy ``+`` ids, prefer the ``-only`` variant, then take the first
        alphabetically. A rerun cannot flip the answer.
        """
        candidates = self.by_url.get(canonical_license_url(uri))
        if not candidates:
            return None, False
        live = [c for c in candidates if not c.get("isDeprecatedLicenseId")] or candidates
        live = [c for c in live if not c["licenseId"].endswith("+")] or live
        ambiguous = len({c["licenseId"] for c in live}) > 1
        only = [c for c in live if c["licenseId"].endswith("-only")]
        return sorted(only or live, key=lambda c: c["licenseId"])[0], ambiguous

    def resolve_name(self, name):
        return self.by_name.get((name or "").strip().lower())


def load_spdx(cache_dir):
    """Fetch the SPDX licence list, caching it beside the DOI responses."""
    path = cache_dir / "spdx-licenses.json"
    if path.is_file():
        return SpdxIndex(json.loads(path.read_text(encoding="utf-8")))
    payload = fetch_json(SPDX_LICENSES_URL)
    if payload is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return SpdxIndex(payload)


def license_block(licence, asserted_uri):
    block = {
        "spdx": licence["licenseId"],
        "name": licence.get("name") or None,
        "url": licence.get("reference") or None,
    }
    if asserted_uri:
        block["sameAs"] = asserted_uri
    return block


def dataset_license(spdx):
    """This repository's own licence, resolved through the SPDX list."""
    if spdx is not None:
        licence, _ = spdx.resolve(DATASET_LICENSE_URI)
        if licence:
            return license_block(licence, DATASET_LICENSE_URI)
    return {"spdx": DATASET_LICENSE_SPDX, "url": DATASET_LICENSE_URI}


def resolve_license(rights_list, spdx):
    """Pick a licence from a DataCite ``rightsList``.

    Not ``rightsList[0]``: ``info:eu-repo/semantics/openAccess`` is an access
    status rather than a licence and is the most frequent entry in this
    databank, so taking the first element would stamp most records with a bogus
    licence. Take the first entry that actually resolves to an SPDX identifier.
    """
    fallback = None
    ambiguous = False
    for entry in rights_list or []:
        uri = (entry.get("rightsUri") or "").strip()
        name = (entry.get("rights") or "").strip()
        if uri.startswith(ACCESS_RIGHTS_PREFIX) or (not uri and not name):
            continue
        if spdx is not None:
            if uri:
                licence, ambiguous = spdx.resolve(uri)
                if licence:
                    return license_block(licence, uri), ambiguous
            licence = spdx.resolve_name(name)
            if licence:
                return license_block(licence, uri or None), False
        if fallback is None:
            # No SPDX match: the asserted URI is the only licence we have, so it
            # goes in `url` where schema.org/license expects it, with a null
            # `spdx` marking it as outside the vocabulary.
            fallback = {"spdx": None, "name": name or None, "url": uri or None}
    return fallback, ambiguous


def access_rights(rights_list):
    """Keep the COAR access status ``resolve_license`` skips, rather than lose it."""
    for entry in rights_list or []:
        uri = (entry.get("rightsUri") or "").strip()
        if uri.startswith(ACCESS_RIGHTS_PREFIX):
            return uri[len(ACCESS_RIGHTS_PREFIX):]
    return None


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------


def normalize_doi(value):
    """Extract a bare DOI from a URL, a ``doi:`` prefix or surrounding prose."""
    match = DOI_RE.search(str(value or ""))
    return match.group(0).rstrip(TRAILING) if match else None


def normalize_orcid(value):
    match = ORCID_RE.search(str(value or ""))
    return f"https://orcid.org/{match.group(1).upper()}" if match else None


def iso_date(value):
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


def clean_text(value):
    """Collapse whitespace in an API-supplied string."""
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value)).strip() or None


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


# ---------------------------------------------------------------------------
# Registry mapping
# ---------------------------------------------------------------------------


def _publisher_name(publisher):
    """DataCite 4.5 allows publisher to be an object rather than a string."""
    return publisher.get("name") if isinstance(publisher, dict) else publisher


def source_block(api, doi, licence, spdx):
    """Provenance for the generated block.

    Records the SPDX list version whenever a licence actually resolved, so a
    surprising identifier can be traced back to the list that produced it --
    SPDX deprecates and re-scopes identifiers between releases.
    """
    source = {"api": api, "doi": doi}
    if spdx is not None and licence and licence.get("spdx"):
        source["spdxLicenseList"] = spdx.version
    return source


def from_datacite(payload, doi, readme, spdx):
    """Map a DataCite record onto Bioschemas properties."""
    attributes = (payload or {}).get("data", {}).get("attributes", {})
    if not attributes:
        return None, []

    notes = []
    titles = attributes.get("titles") or []
    abstract = next(
        (d.get("description") for d in attributes.get("descriptions") or []
         if (d.get("descriptionType") or "") == "Abstract"),
        None,
    )
    dates = {d.get("dateType"): d.get("date") for d in attributes.get("dates") or []}
    published = iso_date(
        dates.get("Issued") or dates.get("Published") or dates.get("Available")
        or attributes.get("publicationYear")
    )

    licence, ambiguous = resolve_license(attributes.get("rightsList"), spdx)
    if ambiguous:
        notes.append("licence URI was ambiguous in the SPDX list")
    if licence is None:
        notes.append("no licence in DataCite rightsList")
    elif licence.get("spdx") is None:
        notes.append(f"licence did not resolve to SPDX: {licence.get('url')}")

    creators = []
    for creator in attributes.get("creators") or []:
        name = clean_text(creator.get("name")) or clean_text(
            " ".join(p for p in (creator.get("givenName"), creator.get("familyName")) if p)
        )
        if not name:
            continue
        entry = {"name": name}
        for identifier in creator.get("nameIdentifiers") or []:
            if (identifier.get("nameIdentifierScheme") or "").upper() == "ORCID":
                orcid = normalize_orcid(identifier.get("nameIdentifier"))
                if orcid:
                    entry["identifier"] = orcid
                    break
        creators.append(entry)
    if not creators:
        notes.append("no creators in DataCite record")

    # PUBLICATION is retired in favour of this list, so existing citations are
    # the authority. Re-deriving solely from PUBLICATION would erase them once
    # that field is gone.
    citations = [str(c) for c in ((readme.get("bioschema_properties") or {}).get("citation") or [])]
    citations += parse_publication(readme.get("PUBLICATION"))
    for related in attributes.get("relatedIdentifiers") or []:
        if (related.get("relationType") in CITATION_RELATIONS
                and (related.get("relatedIdentifierType") or "").upper() == "DOI"):
            citations.append(normalize_doi(related.get("relatedIdentifier")))

    block = {
        "name": clean_text(titles[0].get("title")) if titles else None,
        "description": strip_markup(abstract),
        "datePublished": published,
        "license": licence,
        "publisher": clean_text(_publisher_name(attributes.get("publisher"))),
        "version": clean_text(attributes.get("version")),
        "creator": creators,
        "citation": dedupe(citations),
    }
    rights = access_rights(attributes.get("rightsList"))
    if rights:
        block["accessRights"] = rights
    block["_subjects"] = [s.get("subject") for s in attributes.get("subjects") or [] if s.get("subject")]
    block["_source"] = source_block("datacite", doi, licence, spdx)
    return block, notes


def from_crossref(payload, doi, readme, spdx):
    """Map a CrossRef record onto Bioschemas properties."""
    message = (payload or {}).get("message") or {}
    if not message:
        return None, []

    notes = []
    titles = message.get("title") or []
    issued = (message.get("issued", {}).get("date-parts") or [[]])[0]
    published = iso_date("-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(issued))) if issued else None

    licence, ambiguous = None, False
    entries = message.get("license") or []
    for entry in [e for e in entries if e.get("content-version") == "vor"] or entries:
        uri = (entry.get("URL") or "").strip()
        if not uri:
            continue
        if spdx is not None:
            resolved, ambiguous = spdx.resolve(uri)
            if resolved:
                licence = license_block(resolved, uri)
                break
        if licence is None:
            licence = {"spdx": None, "name": None, "url": uri}
    if licence is None:
        notes.append("no licence in CrossRef record")
    elif licence.get("spdx") is None:
        notes.append(f"licence did not resolve to SPDX: {licence.get('url')}")

    creators = []
    for author in message.get("author") or []:
        name = clean_text(" ".join(p for p in (author.get("given"), author.get("family")) if p)) or clean_text(
            author.get("name")
        )
        if not name:
            continue
        entry = {"name": name}
        orcid = normalize_orcid(author.get("ORCID"))
        if orcid:
            entry["identifier"] = orcid
        creators.append(entry)
    if not creators:
        notes.append("no authors in CrossRef record")

    existing = [str(c) for c in ((readme.get("bioschema_properties") or {}).get("citation") or [])]
    block = {
        "name": clean_text(titles[0]) if titles else None,
        "description": strip_markup(message.get("abstract")),
        "datePublished": published,
        "license": licence,
        "publisher": clean_text(message.get("publisher")),
        "version": None,
        "creator": creators,
        "citation": dedupe(existing + parse_publication(readme.get("PUBLICATION")) + [doi]),
    }
    containers = message.get("container-title") or []
    if containers:
        # Stashed for enrich(): the journal is the parent of the *article*, not
        # of this dataset, so it nests inside isPartOf rather than replacing it.
        block["_journal"] = clean_text(containers[0])
    block["_source"] = source_block("crossref", doi, licence, spdx)
    return block, notes


def with_dataset_license(block, licence):
    """Move a fetched licence to ``articleLicense`` and set the dataset licence.

    Rebuilt rather than mutated so the two stay adjacent and ordered.
    """
    article = block.get("license")
    rebuilt = {}
    for key, value in block.items():
        if key == "license":
            rebuilt["license"] = licence
            if article:
                rebuilt["articleLicense"] = article
        else:
            rebuilt[key] = value
    rebuilt.setdefault("license", licence)
    return rebuilt


# ---------------------------------------------------------------------------
# Local enrichment
# ---------------------------------------------------------------------------


def edam(code, name):
    return {
        "@type": "DefinedTerm",
        "name": name,
        "termCode": code,
        "inDefinedTermSet": EDAM_SET,
        "url": f"{EDAM_SET}/{code}",
    }


def obo(code, name):
    return {
        "@type": "DefinedTerm",
        "name": name,
        "termCode": code,
        "inDefinedTermSet": f"{OBO_SET}/chmo.owl",
        "url": f"{OBO_SET}/{code}",
    }


def data_root_of(path):
    """Walk up from a README.yaml to the databank root."""
    for parent in Path(path).resolve().parents:
        if (parent / "Molecules").is_dir() and (parent / "Simulations").is_dir():
            return parent
    # Both layouts put README.yaml five levels below the root.
    return Path(path).resolve().parents[5]


def record_kind(path):
    return "simulations" if "Simulations" in Path(path).resolve().parts else "experiments"


def experiment_kind(path):
    return "xray" if "FormFactors" in Path(path).resolve().parts else "nmr"


def molecule_names(data_root):
    """Databank molecule id -> chemical name, from ``Molecules/*/*/metadata.yaml``."""
    names = {}
    for meta in Path(data_root).glob("Molecules/*/*/metadata.yaml"):
        try:
            block = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        entry = block.get("NMRlipids") or {}
        if entry.get("name"):
            names[meta.parent.name] = entry["name"]
    return names


def split_subjects(subjects):
    """Zenodo subjects are free text; several pack a list into one string."""
    out = []
    for subject in subjects or []:
        for part in re.split(r"[,;]", str(subject)):
            part = part.strip()
            if part and len(part) < 80:
                out.append(part)
    return out


def dedupe_keywords(entries):
    seen, out = set(), []
    for entry in entries:
        key = (entry["name"] if isinstance(entry, dict) else str(entry)).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(entry)
    return out


def composition_keywords(readme, kind, names):
    if kind == "simulations":
        ids = list((readme.get("COMPOSITION") or {}).keys())
    else:
        ids = list((readme.get("MEMBRANE_COMPOSITION") or {}).keys())
        ids += list((readme.get("SOLUTION_COMPOSITION") or {}).keys())
    words = []
    for mol in ids:
        if mol in KEYWORD_SKIP:
            continue
        words.append(mol)
        if names.get(mol):
            words.append(names[mol])
    return words


def chmo_for_method(readme, path):
    """Map the record onto the most specific CHMO term.

    The directory decides NMR vs X-ray, not the presence of an ``NMR:`` block:
    around twenty OrderParameters records carry no NMR block, and keying off
    that would file them under small-angle X-ray scattering.
    """
    if experiment_kind(path) == "xray":
        return CHMO_SAXS
    method = str((readme.get("NMR") or {}).get("METHOD") or "")
    if method.startswith("PDLF:R"):
        return CHMO_PDLF_R
    if method.startswith("PDLF"):
        return CHMO_PDLF
    if method.startswith(("2H", "CDLF")):
        return CHMO_SSNMR
    return CHMO_NMR


def measurement_technique(readme, path, kind):
    """Ontology terms first, then the free-text detail no term captures."""
    if kind == "simulations":
        out = [edam(*EDAM_MD_OP)]
        engine, version = readme.get("SOFTWARE"), readme.get("SOFTWARE_VERSION")
        if engine:
            out.append(f"Molecular dynamics simulation ({engine}" + (f" {version}" if version else "") + ")")
        if readme.get("FF"):
            out.append(f"{readme['FF']} force field")
        return out

    nmr, xray = readme.get("NMR") or {}, readme.get("XRAY") or {}
    out = [obo(*chmo_for_method(readme, path))]
    if nmr or experiment_kind(path) == "nmr":
        out.append(edam(*EDAM_NMR_TOPIC))
        if nmr.get("METHOD"):
            out.append(f"NMR: {nmr['METHOD']}")
        if nmr.get("INSTRUMENT"):
            out.append(nmr["INSTRUMENT"])
    else:
        out.append(edam(*EDAM_XRAY_TOPIC))
        label = "X-ray scattering"
        if xray.get("SAMPLE_TYPE"):
            label += f" ({xray['SAMPLE_TYPE']})"
        out.append(label)
        if xray.get("SOURCE"):
            out.append(xray["SOURCE"])
    return out


def variables_measured(path, kind):
    """Read off which analyses actually exist beside this README."""
    directory = Path(path).resolve().parent
    found = [label for name, label in VARIABLES.items() if (directory / name).is_file()]
    if any(directory.glob("*OrderParameters.json")):
        found.append("C-H bond order parameter")
    if any(directory.glob("*_FormFactor.json")):
        found.append("X-ray scattering form factor")
    if kind == "experiments" and not found:
        found.append("C-H bond order parameter" if experiment_kind(path) == "nmr"
                     else "X-ray scattering form factor")

    seen, out = set(), []
    for item in found:
        if item in seen:
            continue
        seen.add(item)
        entry = {"@type": "PropertyValue", "name": item}
        code, symbol = UNITS.get(item, (None, None))
        if code:
            entry["unitCode"] = code
        if symbol:
            entry["unitText"] = symbol
        out.append(entry)
    return out


def distribution(readme, path, kind, doi):
    """Where the data actually lives."""
    if kind == "simulations":
        files = []
        for key in ("TRJ", "TPR", "TOP", "GRO", "PSF", "PDB"):
            for group in readme.get(key) or []:
                files += [f for f in (group or []) if isinstance(f, str)]
        entry = {"@type": "DataDownload"}
        if doi:
            entry["contentUrl"] = f"https://doi.org/{doi}"
        entry["name"] = readme.get("SYSTEM") or "trajectory deposition"
        if readme.get("TRAJECTORY_SIZE"):
            entry["contentSize"] = readme["TRAJECTORY_SIZE"]
        for candidate in files:
            suffix = Path(candidate).suffix.lower()
            if suffix in TRAJECTORY_FORMATS:
                entry["encodingFormat"] = TRAJECTORY_FORMATS[suffix]
                break
        if files:
            entry["hasPart"] = files
        return [entry]

    # Experiment data lives in this repository as JSON beside the README. No
    # contentUrl: the public URL is deployment-specific, so the web frontend
    # supplies it.
    directory = Path(path).resolve().parent
    return [
        {"@type": "DataDownload", "name": f.name, "encodingFormat": "application/json"}
        for f in sorted(directory.glob("*.json"))
    ]


def is_based_on(readme, kind):
    """Simulations record which experiment datasets they are compared against."""
    if kind != "simulations":
        return []
    experiment = readme.get("EXPERIMENT") or {}
    refs = [f"experiments/OrderParameters/{e}"
            for entries in (experiment.get("ORDERPARAMETER") or {}).values()
            for e in entries or []]
    refs += [f"experiments/FormFactors/{e}" for e in experiment.get("FORMFACTOR") or []]
    return dedupe(refs)


def part_of(block, readme, kind):
    """What this dataset is part of. Experiments only.

    Preferably the article the values were digitised from, with the journal
    nested one level down as the article's own parent. Where there is no article
    -- the nmrXiv records -- the deposited dataset serves instead. Records with
    neither get nothing rather than an invented parent.
    """
    if kind != "experiments":
        return None
    source_doi = (block.get("_source") or {}).get("doi")

    article = normalize_doi(readme.get("ARTICLE_DOI") or readme.get("DOI"))
    if article:
        entry = {
            "@type": "ScholarlyArticle",
            "@id": f"https://doi.org/{article}",
            "identifier": article,
            "url": f"https://doi.org/{article}",
        }
        if block.get("name") and source_doi == article:
            entry["name"] = block["name"]
        if block.get("_journal"):
            entry["isPartOf"] = {"@type": "Periodical", "name": block["_journal"]}
        return entry

    deposition = normalize_doi(readme.get("DATA_DOI"))
    if deposition:
        entry = {
            "@type": "Dataset",
            "@id": f"https://doi.org/{deposition}",
            "identifier": deposition,
            "url": f"https://doi.org/{deposition}",
        }
        if source_doi == deposition:
            if block.get("name"):
                entry["name"] = block["name"]
            if block.get("publisher"):
                entry["publisher"] = block["publisher"]
        return entry
    return None


def describe(readme, path, kind):
    """A one-line description for records whose registry record has no abstract."""
    if kind == "simulations":
        lipids = [m for m in (readme.get("COMPOSITION") or {}) if m not in KEYWORD_SKIP]
        bits = ["Molecular dynamics simulation of a lipid bilayer"]
        if lipids:
            bits.append("containing " + ", ".join(sorted(lipids)))
        if readme.get("TEMPERATURE"):
            bits.append(f"at {readme['TEMPERATURE']} K")
        tail = []
        if readme.get("FF"):
            tail.append(f"{readme['FF']} force field")
        if readme.get("SOFTWARE"):
            tail.append(str(readme["SOFTWARE"]))
        if readme.get("TRJLENGTH"):
            tail.append(f"{float(readme['TRJLENGTH']) / 1000:.0f} ns trajectory")
        return " ".join(bits) + (", " + ", ".join(tail) + "." if tail else ".")

    membrane = readme.get("MEMBRANE_COMPOSITION") or {}
    what = ("C-H bond order parameters" if experiment_kind(path) == "nmr"
            else "X-ray scattering form factor")
    bits = [f"Experimental {what} for a lipid bilayer"]
    if membrane:
        bits.append("containing " + ", ".join(sorted(membrane)))
    if readme.get("TEMPERATURE"):
        bits.append(f"at {readme['TEMPERATURE']} K")
    technique = [t for t in measurement_technique(readme, path, kind) if isinstance(t, str)]
    return " ".join(bits) + (f", measured by {technique[0]}." if technique else ".")


def enrich(block, readme, path, kind, doi, names):
    """Add every Bioschemas property derivable from local data."""
    subjects = block.pop("_subjects", [])
    if not block.get("description"):
        block["description"] = describe(readme, path, kind)
    if not block.get("name"):
        block["name"] = describe(readme, path, kind)
    if doi:
        # `identifier` is deliberately not written: the web frontend derives it
        # from the record's own DOI field. sameAs keeps the resolvable link.
        block["sameAs"] = f"https://doi.org/{doi}"
    if kind == "simulations" and readme.get("SYSTEM"):
        block["alternateName"] = readme["SYSTEM"]

    terms = EDAM_SIMULATION if kind == "simulations" else (
        EDAM_XRAY if experiment_kind(path) == "xray" else EDAM_NMR
    )
    block["keywords"] = dedupe_keywords(
        [edam(code, label) for code, label in terms]
        + composition_keywords(readme, kind, names)
        + split_subjects(subjects)
    )
    block["measurementTechnique"] = measurement_technique(readme, path, kind)
    block["variableMeasured"] = variables_measured(path, kind)

    dist = distribution(readme, path, kind, doi)
    if dist:
        block["distribution"] = dist
    based = is_based_on(readme, kind)
    if based:
        block["isBasedOn"] = based

    parent = part_of(block, readme, kind)
    block.pop("_journal", None)
    if parent:
        block["isPartOf"] = parent
    else:
        block.pop("isPartOf", None)
    return block


# ---------------------------------------------------------------------------
# Reading and writing README.yaml
# ---------------------------------------------------------------------------


def ordered(block):
    """Canonical key order; anything unlisted keeps its place at the end."""
    out = {k: block[k] for k in BLOCK_ORDER if k in block}
    out.update({k: v for k, v in block.items() if k not in out})
    return out


def prune(block):
    """Drop absent values, as ``Molecules/*/metadata.yaml`` does.

    An omitted property and one set to null mean the same thing in JSON-LD.
    Licences are the exception: an unresolved one keeps an explicit ``spdx:
    null`` beside the URI that failed to resolve, so the gap stays visible.
    """
    pruned = {}
    for key, value in block.items():
        if key in ("license", "articleLicense") and isinstance(value, dict):
            licence = {k: v for k, v in value.items() if v is not None}
            licence.setdefault("spdx", None)
            pruned[key] = licence
        elif value not in (None, [], {}):
            pruned[key] = value
    return pruned


def strip_existing_block(text):
    """Everything above the generated block.

    With no block present the file is returned byte for byte, gaining only the
    newline separating it from what gets appended: several experiment READMEs
    end in blank lines after the deprecated-field section, and normalising those
    away would be an edit to hand-written content.
    """
    index = text.find(LEGACY_SENTINEL)
    if index == -1:
        match = BLOCK_KEY_RE.search(text)
        index = match.start() if match else -1
    if index == -1:
        return text if text.endswith("\n") else text + "\n"
    return text[:index]


def render_block(block):
    return yaml.dump(
        {"bioschema_properties": block},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )


def volatile_stripped(block):
    """A block with ``retrieved`` removed, for change detection.

    Without this the retrieval date alone would make every rerun a diff.
    """
    if not isinstance(block, dict):
        return block
    copy = json.loads(json.dumps(block, sort_keys=True, default=str))
    if isinstance(copy.get("_source"), dict):
        copy["_source"].pop("retrieved", None)
    return copy


# Date fields normalised per record kind, mapped to whether a ``DD/MM/YYYY``
# value may be reinterpreted as ISO.
#
# Only DATEOFRUNNING may: add_simulation.py writes it with strftime("%d/%m/%Y"),
# so the convention is known. FF_DATE is hand-entered and the corpus proves both
# conventions are in use -- 14/10/2025 and 26/01/2021 can only be DD/MM, while
# 3/22/21 can only be MM/DD -- alongside values like "?/?/2020" and "10/2023".
# Reinterpreting those would silently corrupt them, so slash-formatted values
# are left exactly as they are; both fields still get quoted when already ISO,
# which is what the schemas need (unquoted, YAML yields a date object).
#
# A simulation's DATE is not declared in readme_yaml_schema.json at all, so it
# is left as YAML parses it rather than rewritten for tidiness.
DATE_FIELDS = {
    "simulations": {"DATEOFRUNNING": True, "FF_DATE": False},
    "experiments": {"DATE": False},
}


def normalize_dates(text, kind):
    """Rewrite the record's date fields as quoted ISO strings.

    Quoting matters: unquoted, YAML parses ``2011-07-26`` into a date object and
    the schemas require these fields to be strings. ``DATEOFRUNNING`` also
    arrives as ``DD/MM/YYYY`` from ``add_simulation.py``.
    """
    fields = "|".join(DATE_FIELDS[kind])
    lines = text.splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines):
        match = re.match(rf"^({fields}):[ \t]*['\"]?([^'\"#\n]+?)['\"]?[ \t]*(#.*)?$",
                         line.rstrip("\n"))
        if not match:
            continue
        key, raw, comment = match.group(1), match.group(2).strip(), match.group(3)
        slashed = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", raw)
        if slashed and DATE_FIELDS[kind][key]:
            value = f"{slashed.group(3)}-{slashed.group(2)}-{slashed.group(1)}"
        else:
            iso = re.match(r"^(\d{4}-\d{2}-\d{2})", raw)
            if not iso:
                continue
            value = iso.group(1)
        new = f"{key}: '{value}'" + (f"  {comment}" if comment else "")
        new += "\n" if line.endswith("\n") else ""
        if new != line:
            lines[i] = new
            changed = True
    return "".join(lines), changed


def drop_publication(text, block):
    """Retire ``PUBLICATION`` once ``citation`` demonstrably covers it.

    Refuses to remove anything the citation list does not already carry, so a
    hand-written reference cannot be lost to a parsing slip.
    """
    doc = yaml.safe_load(text) or {}
    if "PUBLICATION" not in doc:
        return text, False
    cites = [str(c) for c in (block.get("citation") or [])]
    for item in parse_publication(doc.get("PUBLICATION")):
        if item not in cites:
            print(f"  keeping PUBLICATION: {item!r} is not in citation")
            return text, False
    kept = [ln for ln in text.splitlines(keepends=True) if not ln.startswith("PUBLICATION:")]
    return "".join(kept), True


def resolve_doi(doi, kind, cache_dir):
    """Fetch one DOI, memoised on disk. Returns ``(payload, api)``.

    Simulations are Zenodo depositions, so DataCite answers. Experiments are
    usually journal articles, so CrossRef answers -- but a few cite an nmrXiv or
    Zenodo deposition instead, and those are registered with DataCite, so fall
    back to it rather than skipping them.
    """
    slug = urllib.parse.quote(doi, safe="")
    cached = cache_dir / f"{slug}.json"
    if cached.is_file():
        payload, api = json.loads(cached.read_text(encoding="utf-8"))
        return payload, api

    quoted = urllib.parse.quote(doi, safe="/")
    order = ([("datacite", DATACITE_URL)] if kind == "simulations"
             else [("crossref", CROSSREF_URL), ("datacite", DATACITE_URL)])
    payload, api = None, order[0][0]
    for candidate, template in order:
        payload = fetch_json(template.format(doi=quoted))
        if payload is not None:
            api = candidate
            break

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps([payload, api]), encoding="utf-8")
    return payload, api


def process(path, spdx, names, cache_dir, dry_run=False):
    """Enrich one README.yaml in place. Returns True when the file changed."""
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    readme = yaml.safe_load(original) or {}
    kind = record_kind(path)

    if kind == "simulations":
        doi = normalize_doi(readme.get("DOI"))
    else:
        # 29 experiments still carry the deprecated `DOI` key. normalize_doi
        # returns None for the `unpublished/<slug>` values it also holds.
        doi = normalize_doi(readme.get("ARTICLE_DOI") or readme.get("DATA_DOI") or readme.get("DOI"))

    block, notes = None, []
    if doi:
        payload, api = resolve_doi(doi, kind, cache_dir)
        if payload is None:
            print(f"  warning: DOI not found: {doi}")
        else:
            mapper = from_datacite if api == "datacite" else from_crossref
            block, notes = mapper(payload, doi, readme, spdx)
    if block is None:
        # unpublished/<slug> records have nothing to look up, but they are still
        # data this repository distributes, so they carry the dataset licence.
        block = {"_source": {"api": None, "doi": None}}
    for note in notes:
        print(f"  note: {note}")

    block = enrich(block, readme, path, kind, doi, names)
    if kind == "experiments":
        block = with_dataset_license(block, dataset_license(spdx))
    block = ordered(prune(block))

    text, dates_changed = normalize_dates(original, kind)
    if kind == "simulations":
        text, _ = drop_publication(text, block)

    existing = readme.get("bioschema_properties")
    if volatile_stripped(existing) == volatile_stripped(block):
        block.setdefault("_source", {})["retrieved"] = (
            ((existing or {}).get("_source") or {}).get("retrieved") or date.today().isoformat()
        )
    else:
        block.setdefault("_source", {})["retrieved"] = date.today().isoformat()

    updated = strip_existing_block(text) + render_block(block)
    if updated == original:
        return False
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    print(f"{'Would update' if dry_run else 'Updated'} bioschema metadata in {path}"
          + (" (dates normalised)" if dates_changed else ""))
    return True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check(path, spdx, strict=False):
    """Validate a generated block. Returns ``(level, message)`` pairs.

    ERROR means the block is unusable; WARNING means it is well-formed but
    incomplete. Two things warn rather than fail: a licence outside SPDX is the
    expected outcome for experiments, whose CrossRef licences are publisher
    text-mining terms, and a missing block is normal for a freshly uploaded
    record the workflow has not reached yet.
    """
    readme = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    block = readme.get("bioschema_properties")
    level = "ERROR" if strict else "WARNING"
    if block is None:
        return [(level, f"{level}: {path}: missing bioschema_properties block")]
    if not isinstance(block, dict):
        return [("ERROR", f"ERROR: {path}: bioschema_properties is not a mapping")]

    found = []

    def error(message):
        found.append(("ERROR", f"ERROR: {path}: {message}"))

    def warn(message):
        found.append(("WARNING", f"WARNING: {path}: {message}"))

    resolved = (block.get("_source") or {}).get("doi") is not None
    if resolved and not clean_text(block.get("name")):
        error("name is empty")

    published = block.get("datePublished")
    if resolved and published is None:
        error("datePublished is missing")
    elif published is not None and not DATE_RE.match(str(published)):
        error(f"datePublished {published!r} is not YYYY[-MM[-DD]]")

    licence = block.get("license")
    if not isinstance(licence, dict):
        warn("no licence: the DOI record declares none")
    elif licence.get("spdx") is None:
        warn(f"licence did not resolve to SPDX (url {licence.get('url')!r})")
    elif spdx is not None and licence["spdx"] not in spdx.by_id:
        error(f"license.spdx {licence['spdx']!r} is not in the SPDX list")

    article = block.get("articleLicense")
    if isinstance(article, dict) and article.get("spdx") is None:
        warn(f"articleLicense is outside SPDX -- publisher terms ({article.get('url')!r})")

    creators = block.get("creator")
    if resolved and not creators:
        error("creator is empty")
    for index, creator in enumerate(creators or []):
        if not isinstance(creator, dict) or not clean_text(creator.get("name")):
            error(f"creator[{index}] has no name")
        elif creator.get("identifier") and not ORCID_RE.search(str(creator["identifier"])):
            error(f"creator[{index}].identifier {creator['identifier']!r} is not an ORCID URI")

    return found


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="README.yaml paths to enrich")
    parser.add_argument("--cache", type=Path, default=None,
                        help="directory for cached API responses")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument("--check", action="store_true",
                        help="validate existing blocks instead of writing")
    parser.add_argument("--strict", action="store_true",
                        help="with --check, fail on a missing block rather than warn")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"error: no such file: {missing[0]}")

    root = data_root_of(paths[0])
    cache_dir = args.cache or Path(os.environ.get("RUNNER_TEMP", root)) / ".cache" / "bioschema"
    spdx = load_spdx(cache_dir)
    if spdx is None:
        print("warning: SPDX licence list unavailable; licences will not resolve")

    if args.check:
        failed = set()
        for path in paths:
            for level, message in check(path, spdx, args.strict):
                print(message)
                if level == "ERROR":
                    failed.add(path)
        print(f"\n{len(paths) - len(failed)} of {len(paths)} records valid"
              + (f", {len(failed)} failing" if failed else ""))
        sys.exit(1 if failed else 0)

    names = molecule_names(root)
    changed = sum(process(p, spdx, names, cache_dir, args.dry_run) for p in paths)
    print(f"\n{changed} of {len(paths)} records {'would be ' if args.dry_run else ''}updated")


if __name__ == "__main__":
    main()

"""
Reading and rewriting a ``README.yaml``.

The file is never re-serialised: the generated block is appended and only that
block is rewritten, so the hand-written comments most experiment records carry
survive a run. What is normalised beyond the block are the date fields, which
YAML would otherwise parse into date objects the schemas reject.

``process`` is one record end to end -- read, resolve the DOI, enrich, write --
and is what the command line maps over the databank.
"""

import json
import re
from datetime import date
from pathlib import Path

import yaml

from .constants import BLOCK_KEY_RE, BLOCK_ORDER, DATE_FIELDS, LEGACY_SENTINEL
from .bioschema import enrich
from .fields import record_dois, record_kind
from .helpers import parse_publication
from .licenses import dataset_license
from .registries import from_crossref, from_datacite, resolve_doi, with_dataset_license


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


def process(path, spdx, names, cache_dir, dry_run=False):
    """Enrich one README.yaml in place. Returns True when the file changed."""
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    readme = yaml.safe_load(original) or {}
    kind = record_kind(path)

    dois = record_dois(readme, kind)
    doi = dois.lookup

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

    block = enrich(block, readme, path, kind, dois, names)
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

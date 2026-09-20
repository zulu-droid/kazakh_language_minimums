"""
Extract Kazakh / Russian / English vocabulary triples from the
"Leksikalyk minimum" PDF series (A1, A2, B1, B2, C1) into a single
Unicode CSV file.

Each PDF renders its word list as a 3-column table (Kazakh | Russian |
English), even though the on-page shapes differ from level to level:
A1/A2/B1 use a dense phrase/word table (row height set by whichever
column wraps the most), while B2/C1 use topic-grouped word lists (a
header then a long plain list of words per column). PyMuPDF's own
paragraph/"block" detection sometimes merges neighbouring columns or
rows together, so this script works from raw per-line bounding boxes
instead: it re-clusters lines into 3 x-position columns, then
reconstructs rows with one of two strategies depending on the level
(see compute_row_threshold/group_rows vs. group_rows_by_agreement).

The "МАҚАЛ-МӘТЕЛДЕР" (proverbs/sayings) section at the end of several
levels is free-flowing text rather than a strict word-for-word table
(Kazakh/Russian/English wrap independently, at very different lengths),
so its rows are lower quality than the rest - occasionally truncated or
merged - but are still included per request.

Output columns: Id, Kazakh, Russian, English, Level.
Dedup rule: if the same Kazakh word appears in more than one level, only
the first level it appears in (processed in A1, A2, B1, B2, C1 order) is
kept.
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

LEVELS = ["A1", "A2", "B1", "B2", "C1"]
SRC_DIR = Path(r"C:\temp\Kazakh_Language")
FILES = {lvl: SRC_DIR / f"\u041b\u0435\u043a\u0441\u0438\u043a\u0430\u043b\u044b\u049b_\u043c\u0438\u043d\u0438\u043c\u0443\u043c_{lvl}.pdf" for lvl in LEVELS}
OUT_CSV = SRC_DIR / "kazakh_lexical_minimum.csv"

PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
BOLIM_RE = re.compile(r"\u0411\u04e8\u041b\u0406\u041c")  # BOLIM (section marker word)

COLUMN_GAP = 40       # min x-gap (pt) that separates two columns
COLUMN_TOLERANCE = 55  # max distance (pt) from a column centroid to still belong to it
MIN_COLUMN_POPULATION = 4  # a page needs at least this many lines per column to count as tabular


def get_lines(page):
    d = page.get_text("dict")
    lines = []
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append({"x0": x0, "y0": y0, "y1": y1, "text": text})
    return lines


def is_bilingual_banner(text):
    """Page/section banners are sometimes rendered as a single line
    mixing two languages, e.g. "ПРИРОДА, NATURE" (a Cyrillic title next
    to its English title, comma-joined). A real table cell only ever
    holds one language, so this pattern is a reliable, narrow signal
    that a line is a banner rather than vocabulary - unlike a blanket
    "is uppercase" filter, it never risks dropping a real short
    all-caps entry (e.g. an English abbreviation)."""
    has_cyrillic = any("CYRILLIC" in unicodedata.name(c, "") for c in text if c.isalpha())
    has_latin = any("LATIN" in unicodedata.name(c, "") for c in text if c.isalpha())
    return "," in text and has_cyrillic and has_latin


def prefilter(lines):
    out = []
    for ln in lines:
        t = ln["text"]
        if PAGE_NUM_RE.match(t):
            continue
        if BOLIM_RE.search(t):
            continue
        if is_bilingual_banner(t):
            continue
        out.append(ln)
    return out


def cluster_columns(lines):
    """Return 3 (lo, hi) x-ranges for the kk/ru/en columns, or None if this
    page doesn't look like a 3-column vocabulary table."""
    if len(lines) < MIN_COLUMN_POPULATION * 3:
        return None

    xs = sorted(l["x0"] for l in lines)
    clusters = []
    current = [xs[0]]
    for x in xs[1:]:
        if x - current[-1] > COLUMN_GAP:
            clusters.append(current)
            current = [x]
        else:
            current.append(x)
    clusters.append(current)

    ranges = [(min(c), max(c), len(c)) for c in clusters]
    ranges.sort(key=lambda r: -r[2])
    top3 = ranges[:3]
    if len(top3) < 3 or min(r[2] for r in top3) < MIN_COLUMN_POPULATION:
        return None
    top3.sort(key=lambda r: r[0])
    return [(lo, hi) for lo, hi, _ in top3]


def assign_columns(lines, ranges):
    centroids = [(lo + hi) / 2 for lo, hi in ranges]
    cols = [[], [], []]
    for ln in lines:
        best_i, best_d = None, None
        for i, c in enumerate(centroids):
            d = abs(ln["x0"] - c)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_d <= COLUMN_TOLERANCE:
            cols[best_i].append(ln)
    return cols


def compute_row_threshold(cols):
    """Figure out, from this page's own line spacing, the y0-to-y0 delta
    that separates a wrapped continuation line from the start of a new
    row. Wrapped lines sit tighter together (plain line leading) than
    two independent rows (leading + paragraph spacing), so on a page
    that actually has wrapped cells the deltas form two clusters and we
    split at the first real jump above the tight-wrap cluster.

    Pages with no wrapped cells at all (e.g. a dense single-word list)
    have no such second cluster - their deltas are all "new row" even
    though they can be smaller than another page's wrap cluster. Merging
    would be catastrophic there (it would collapse dozens of real
    entries into one), so we only accept a split when both sides of it
    are well populated; otherwise we assume there is no wrapping on this
    page at all and never merge."""
    deltas = []
    for col in cols:
        cs = sorted(col, key=lambda l: l["y0"])
        for i in range(1, len(cs)):
            d = cs[i]["y0"] - cs[i - 1]["y0"]
            if 2 < d < 60:
                deltas.append(d)
    if len(deltas) < 10:
        return 0.0
    deltas.sort()
    for i in range(1, len(deltas)):
        if deltas[i] - deltas[i - 1] > 2.5:
            low_count = i
            high_count = len(deltas) - i
            if low_count / len(deltas) >= 0.10 and high_count / len(deltas) >= 0.10:
                return (deltas[i] + deltas[i - 1]) / 2
            return 0.0
    return 0.0


def group_rows(col_lines, threshold):
    """Group a column's lines (already sorted by y0) into row-cells,
    merging wrapped continuation lines."""
    col_lines = sorted(col_lines, key=lambda l: l["y0"])
    cells = []
    prev_y0 = None
    for ln in col_lines:
        if cells and prev_y0 is not None and ln["y0"] - prev_y0 <= threshold:
            cells[-1]["text"] += " " + ln["text"]
            cells[-1]["y1"] = max(cells[-1]["y1"], ln["y1"])
        else:
            cells.append({"y0": ln["y0"], "y1": ln["y1"], "text": ln["text"]})
        prev_y0 = ln["y0"]
    return cells


def group_rows_by_agreement(cols, y_tol=2.0):
    """Alternative row-reconstruction used for the "grouped list" layout
    (B2/C1 levels): a topic header is followed by a long, plain list of
    words per column, with no extra paragraph spacing between rows at
    all - so geometric gap thresholds can't tell a wrapped continuation
    line from the next word (both are ~1 line-height apart).

    Instead this uses the one thing that IS reliable in that layout: a
    genuine new row always starts at (almost) the same y as the other
    two columns' next row, because the table row height is synced
    across columns. A y-position that only ONE column has a line at is
    therefore a wrapped continuation of that column's current cell, not
    a new row.

    (This does not work for the dense phrase tables used by A1/A2/B1,
    where a long phrase can wrap all three columns together at the same
    y - there, a wrap and a new row both look like every column
    agreeing, so extract_pdf() uses group_rows()/compute_row_threshold
    for those instead.)"""
    tagged = []
    for ci, col in enumerate(cols):
        for ln in col:
            tagged.append({**ln, "col": ci})
    tagged.sort(key=lambda l: l["y0"])

    buckets = []
    for ln in tagged:
        if buckets and ln["y0"] - buckets[-1][-1]["y0"] <= y_tol:
            buckets[-1].append(ln)
        else:
            buckets.append([ln])

    row_start = {0} if buckets else set()
    for i, b in enumerate(buckets):
        if len({ln["col"] for ln in b}) >= 2:
            row_start.add(i)

    rows = []
    current = {0: None, 1: None, 2: None}
    for i, b in enumerate(buckets):
        if i in row_start and any(v is not None for v in current.values()):
            rows.append(current)
            current = {0: None, 1: None, 2: None}
        for ln in b:
            c = ln["col"]
            if current[c] is None:
                current[c] = {"y0": ln["y0"], "y1": ln["y1"], "text": ln["text"]}
            else:
                current[c]["text"] += " " + ln["text"]
                current[c]["y1"] = max(current[c]["y1"], ln["y1"])
    if any(v is not None for v in current.values()):
        rows.append(current)

    complete = []
    for r in rows:
        if all(r[c] is not None for c in range(3)):
            complete.append((r[0], r[1], r[2]))
    return complete


def match_rows(kk_cells, ru_cells, en_cells, warnings, page_no):
    """Pair up cells across the 3 columns. Prefer simple index alignment
    (rows share a synced height across columns); fall back to nearest-y
    matching when counts disagree."""
    if len(kk_cells) == len(ru_cells) == len(en_cells):
        return list(zip(kk_cells, ru_cells, en_cells))

    warnings.append(
        f"page {page_no}: column counts differ (kk={len(kk_cells)}, "
        f"ru={len(ru_cells)}, en={len(en_cells)}) - using nearest-y matching"
    )
    rows = []
    used_ru, used_en = set(), set()
    for kk in kk_cells:
        ru_match = min(
            (r for i, r in enumerate(ru_cells) if i not in used_ru),
            key=lambda r: abs(r["y0"] - kk["y0"]),
            default=None,
        )
        en_match = min(
            (e for i, e in enumerate(en_cells) if i not in used_en),
            key=lambda e: abs(e["y0"] - kk["y0"]),
            default=None,
        )
        tol = 20
        if ru_match is not None and abs(ru_match["y0"] - kk["y0"]) > tol:
            ru_match = None
        if en_match is not None and abs(en_match["y0"] - kk["y0"]) > tol:
            en_match = None
        if ru_match is None or en_match is None:
            warnings.append(f"page {page_no}: dropped unmatched row near y={kk['y0']:.0f} kk={kk['text']!r}")
            continue
        used_ru.add(ru_cells.index(ru_match))
        used_en.add(en_cells.index(en_match))
        rows.append((kk, ru_match, en_match))
    return rows


def is_banner(text):
    """All-caps rows are section/topic banners duplicated elsewhere in
    lowercase, not real vocabulary entries. Ignore single characters
    (e.g. the pronoun 'I') so we never misclassify a real short entry."""
    if len(text) <= 1:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(unicodedata.category(c).startswith("Lu") for c in letters)


# A1/A2/B1 use a dense phrase/word table where wrapped lines are
# genuinely tighter-spaced than new rows, so the geometric threshold
# approach works. B2/C1 use topic-grouped word lists with uniform
# line spacing throughout, where only cross-column agreement can tell
# a new row from a wrapped continuation.
AGREEMENT_LEVELS = {"B2", "C1"}


def extract_pdf(path, level, warnings):
    doc = pymupdf.open(path)
    entries = []
    tabular_pages = 0
    for page_no in range(len(doc)):
        page = doc[page_no]
        lines = prefilter(get_lines(page))
        ranges = cluster_columns(lines)
        if ranges is None:
            continue
        tabular_pages += 1
        cols = assign_columns(lines, ranges)

        if level in AGREEMENT_LEVELS:
            rows = group_rows_by_agreement(cols)
        else:
            threshold = compute_row_threshold(cols)
            kk_cells = group_rows(cols[0], threshold)
            ru_cells = group_rows(cols[1], threshold)
            en_cells = group_rows(cols[2], threshold)
            rows = match_rows(kk_cells, ru_cells, en_cells, warnings, page_no)

        for kk, ru, en in rows:
            kk_t, ru_t, en_t = kk["text"].strip(), ru["text"].strip(), en["text"].strip()
            if is_banner(kk_t) and is_banner(ru_t) and is_banner(en_t):
                continue
            if not kk_t or not ru_t or not en_t:
                continue
            entries.append((kk_t, ru_t, en_t))
    print(f"  {level}: {len(doc)} pages, {tabular_pages} tabular pages, {len(entries)} raw entries")
    return entries


def main():
    seen = {}
    ordered = []
    all_warnings = []

    for level in LEVELS:
        path = FILES[level]
        if not path.exists():
            print(f"missing file: {path}", file=sys.stderr)
            continue
        warnings = []
        entries = extract_pdf(path, level, warnings)
        all_warnings.extend(f"[{level}] {w}" for w in warnings)

        new_count = 0
        for kk, ru, en in entries:
            key = " ".join(kk.split()).casefold()
            if key in seen:
                continue
            seen[key] = True
            ordered.append((kk, ru, en, level))
            new_count += 1
        print(f"  {level}: {new_count} new unique words after dedup")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Kazakh", "Russian", "English", "Level"])
        for i, (kk, ru, en, level) in enumerate(ordered, start=1):
            writer.writerow([i, kk, ru, en, level])

    print(f"\nWrote {len(ordered)} rows to {OUT_CSV}")

    if all_warnings:
        warn_path = SRC_DIR / "extract_lexicon_warnings.log"
        with open(warn_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_warnings))
        print(f"{len(all_warnings)} warnings written to {warn_path}")


if __name__ == "__main__":
    main()

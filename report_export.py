"""
DOR-4: export the forecast reports of a run into human readable files.

For every run a folder is created under reports/ with:
- osszefoglalo.csv : one row per question (opens in Excel / Google Sheets)
- riport.html      : readable page with the forecast, research and reasoning

The export must never break the bot: every error is caught and logged.
Hungarian labels are kept in the LABELS dict as unicode escapes, so this
source file stays ASCII only.
"""

from __future__ import annotations

import csv
import html
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Sequence

from forecasting_tools import (
    BinaryReport,
    ForecastReport,
    MultipleChoiceReport,
    NumericDistribution,
    NumericReport,
)

logger = logging.getLogger(__name__)

LABELS = {
    "title": "dorka riport",
    "question": "K\u00e9rd\u00e9s",
    "link": "Link",
    "type": "T\u00edpus",
    "forecast": "dorka tippje",
    "community": "K\u00f6z\u00f6ss\u00e9gi tipp",
    "deviation": "Elt\u00e9r\u00e9s (sz\u00e1zal\u00e9kpont)",
    "errors": "Kiolvas\u00e1si hib\u00e1k",
    "cost": "K\u00f6lts\u00e9g (USD)",
    "minutes": "Id\u0151 (perc)",
    "summary_table": "\u00d6sszefoglal\u00f3",
    "failed": "Hib\u00e1val elakadt k\u00e9rd\u00e9sek",
    "research": "Kutat\u00e1s",
    "rationales": "Az el\u0151rejelz\u0151k indokl\u00e1sa",
    "error_details": "Hib\u00e1k r\u00e9szletei",
    "run": "Fut\u00e1s",
    "mode": "M\u00f3d",
    "count": "Sikeres k\u00e9rd\u00e9sek",
    "total_cost": "\u00d6sszk\u00f6lts\u00e9g (USD)",
    "below_lower": "als\u00f3 hat\u00e1r alatt",
    "above_upper": "fels\u0151 hat\u00e1r felett",
    "before_today": "mai nap el\u0151tt",
    "median": "medi\u00e1n",
    "na": "\u2013",
    "binary": "bin\u00e1ris",
    "multiple_choice": "feleletv\u00e1laszt\u00f3s",
    "numeric": "numerikus",
    "date": "d\u00e1tum",
    "discrete": "diszkr\u00e9t",
    "conditional": "felt\u00e9teles",
    "other": "egy\u00e9b",
}


def export_reports(
    reports: Sequence[ForecastReport | BaseException],
    base_folder: str,
    run_mode: str,
) -> str | None:
    """Writes the CSV and HTML files. Returns the run folder, or None on error."""
    if not reports:
        logger.info("No reports in this run, nothing to export")
        return None
    try:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(base_folder, f"run-{run_id}")
        os.makedirs(folder, exist_ok=True)

        good = [r for r in reports if isinstance(r, ForecastReport)]
        failed = [r for r in reports if isinstance(r, BaseException)]
        rows = [_build_row(r) for r in good]

        _write_csv(os.path.join(folder, "osszefoglalo.csv"), rows)
        _write_html(
            os.path.join(folder, "riport.html"), good, rows, failed, run_id, run_mode
        )
        logger.info(f"Report export written to {folder}")
        return folder
    except Exception as e:  # never break the bot because of the export
        logger.error(f"Report export failed: {e.__class__.__name__}: {e}")
        return None


# ---------------------------------------------------------------- rows


def _build_row(report: ForecastReport) -> dict[str, Any]:
    question = report.question
    row: dict[str, Any] = {
        "question": getattr(question, "question_text", ""),
        "url": getattr(question, "page_url", ""),
        "type": _type_label(report),
        "forecast": LABELS["na"],
        "community": LABELS["na"],
        "deviation": LABELS["na"],
        "errors": len(report.errors or []),
        "cost": _round(report.price_estimate, 4),
        "minutes": _round(report.minutes_taken, 1),
    }
    try:
        row["forecast"] = _readable_forecast(report)
    except Exception as e:
        row["forecast"] = f"? ({e.__class__.__name__})"
    if isinstance(report, BinaryReport):
        try:
            community = report.community_prediction
            if community is not None:
                row["community"] = f"{community * 100:.1f}%"
                row["deviation"] = f"{abs(report.prediction - community) * 100:.1f}"
        except Exception:
            pass
    return row


def _type_label(report: ForecastReport) -> str:
    name = report.__class__.__name__
    mapping = {
        "BinaryReport": "binary",
        "MultipleChoiceReport": "multiple_choice",
        "NumericReport": "numeric",
        "DateReport": "date",
        "DiscreteReport": "discrete",
        "ConditionalReport": "conditional",
    }
    return LABELS[mapping.get(name, "other")]


def _readable_forecast(report: ForecastReport) -> str:
    prediction = report.prediction
    if isinstance(report, BinaryReport):
        return f"{prediction * 100:.1f}%"
    if isinstance(report, MultipleChoiceReport):
        options = sorted(
            prediction.predicted_options, key=lambda o: o.probability, reverse=True
        )
        return "; ".join(f"{o.option_name}: {o.probability * 100:.1f}%" for o in options)
    if isinstance(report, NumericReport) and isinstance(prediction, NumericDistribution):
        return _readable_distribution(prediction)
    return str(prediction)[:300]


def _readable_distribution(dist: NumericDistribution) -> str:
    # DOR-17: the library report prints percentiles beyond an open bound as the
    # bound itself. Here we show the real probability mass outside the bounds.
    cdf = dist.get_cdf()
    below_lower = cdf[0].percentile
    above_upper = 1 - cdf[-1].percentile
    median = _value_at(cdf, 0.5)
    parts = [f"{LABELS['median']}: {_fmt_value(median, dist)}"]
    if dist.is_date:
        now = datetime.now(timezone.utc).timestamp()
        before_today = max(
            [p.percentile for p in cdf if p.value <= now], default=below_lower
        )
        if dist.lower_bound < now:
            parts.append(f"{LABELS['before_today']}: {before_today * 100:.1f}%")
    if dist.open_lower_bound or below_lower > 0.0005:
        parts.append(f"{LABELS['below_lower']}: {below_lower * 100:.1f}%")
    if dist.open_upper_bound or above_upper > 0.0005:
        parts.append(f"{LABELS['above_upper']}: {above_upper * 100:.1f}%")
    return "; ".join(parts)


def _value_at(cdf: list, target: float) -> float | None:
    """Value where the cdf reaches target, None if it is outside the range."""
    if target <= cdf[0].percentile or target > cdf[-1].percentile:
        return None
    for prev, cur in zip(cdf, cdf[1:]):
        if prev.percentile < target <= cur.percentile:
            span = cur.percentile - prev.percentile
            ratio = 0 if span == 0 else (target - prev.percentile) / span
            return prev.value + ratio * (cur.value - prev.value)
    return None


def _fmt_value(value: float | None, dist: NumericDistribution) -> str:
    if value is None:
        # the median is outside the question range
        if dist.get_cdf()[-1].percentile < 0.5:
            return f"> {_fmt_plain(dist.upper_bound, dist)}"
        return f"< {_fmt_plain(dist.lower_bound, dist)}"
    return _fmt_plain(value, dist)


def _fmt_plain(value: float, dist: NumericDistribution) -> str:
    if dist.is_date:
        return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d")
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.3g}"


def _round(value: float | None, digits: int) -> Any:
    return LABELS["na"] if value is None else round(value, digits)


# ---------------------------------------------------------------- csv


CSV_COLUMNS = [
    "question",
    "url",
    "type",
    "forecast",
    "community",
    "deviation",
    "errors",
    "cost",
    "minutes",
]


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    header = [LABELS.get(c, c) if c != "url" else LABELS["link"] for c in CSV_COLUMNS]
    # utf-8-sig: Excel then shows the accented letters correctly
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[c] for c in CSV_COLUMNS])


# ---------------------------------------------------------------- html


def _markdown_to_html(text: str) -> str:
    """Small markdown renderer for the parts the reports use (no extra
    dependency): headings, bold, italic, links, bullet lists, paragraphs.
    Everything is HTML-escaped first, so no raw HTML from the LLM gets through."""
    lines = html.escape(text or "").split("\n")
    out: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def inline(t: str) -> str:
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
        t = re.sub(
            r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<a href='\2'>\1</a>", t
        )
        return t

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + "<br>".join(inline(x) for x in paragraph) + "</p>")
            paragraph.clear()

    for raw in lines:
        line = raw.rstrip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if heading or bullet or not line.strip():
            flush_paragraph()
        if in_list and not bullet:
            out.append("</ul>")
            in_list = False
        if heading:
            level = min(len(heading.group(1)) + 2, 6)
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
        elif bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(bullet.group(1))}</li>")
        elif line.strip():
            paragraph.append(line)
    flush_paragraph()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _section(report: ForecastReport, name: str) -> str:
    try:
        return getattr(report, name)
    except Exception:
        return ""


STYLE = """
body { font-family: Lato, Calibri, "Segoe UI", Arial, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; color: #172b4d; line-height: 1.5; }
h1 { margin-bottom: 0.2rem; }
.meta { color: #5e6c84; margin-bottom: 1.5rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th, td { border: 1px solid #dfe1e6; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f4f5f7; }
.q { border: 1px solid #dfe1e6; border-radius: 6px; padding: 1rem 1.2rem; margin: 1.5rem 0; }
.q h2 { font-size: 1.15rem; margin: 0 0 0.4rem 0; }
.forecast { background: #e3fcef; padding: 0.5rem 0.8rem; border-radius: 4px; }
.err { background: #ffebe6; padding: 0.5rem 0.8rem; border-radius: 4px; }
details { margin: 0.6rem 0; }
summary { cursor: pointer; font-weight: bold; }
pre { white-space: pre-wrap; }
"""


def _write_html(
    path: str,
    reports: list[ForecastReport],
    rows: list[dict[str, Any]],
    failed: list[BaseException],
    run_id: str,
    run_mode: str,
) -> None:
    e = html.escape
    total_cost = sum(
        r.price_estimate for r in reports if isinstance(r.price_estimate, (int, float))
    )
    out: list[str] = [
        "<!DOCTYPE html><html lang='hu'><head><meta charset='utf-8'>",
        f"<title>{e(LABELS['title'])} {e(run_id)}</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<h1>{e(LABELS['title'])}</h1>",
        "<div class='meta'>",
        f"{e(LABELS['run'])}: {e(run_id)} UTC &middot; {e(LABELS['mode'])}: {e(run_mode)}"
        f" &middot; {e(LABELS['count'])}: {len(reports)}"
        f" &middot; {e(LABELS['total_cost'])}: {total_cost:.3f}",
        "</div>",
        f"<h2>{e(LABELS['summary_table'])}</h2>",
        "<table><tr>",
    ]
    for col in ["question", "type", "forecast", "community", "deviation", "errors", "cost"]:
        out.append(f"<th>{e(LABELS[col])}</th>")
    out.append("</tr>")
    for i, row in enumerate(rows):
        out.append(
            f"<tr><td><a href='#q{i}'>{e(str(row['question']))}</a></td>"
            f"<td>{e(str(row['type']))}</td><td>{e(str(row['forecast']))}</td>"
            f"<td>{e(str(row['community']))}</td><td>{e(str(row['deviation']))}</td>"
            f"<td>{e(str(row['errors']))}</td><td>{e(str(row['cost']))}</td></tr>"
        )
    out.append("</table>")

    if failed:
        out.append(f"<h2>{e(LABELS['failed'])}</h2><div class='err'><ul>")
        for exc in failed:
            out.append(f"<li>{e(exc.__class__.__name__)}: {e(str(exc)[:500])}</li>")
        out.append("</ul></div>")

    for i, (report, row) in enumerate(zip(reports, rows)):
        out.append(f"<div class='q' id='q{i}'>")
        out.append(f"<h2>{e(str(row['question']))}</h2>")
        if row["url"]:
            out.append(f"<p><a href='{e(str(row['url']))}'>{e(str(row['url']))}</a></p>")
        out.append(
            f"<p class='forecast'><b>{e(LABELS['forecast'])}:</b> {e(str(row['forecast']))}"
            f" &middot; <b>{e(LABELS['community'])}:</b> {e(str(row['community']))}</p>"
        )
        if report.errors:
            out.append(f"<details class='err'><summary>{e(LABELS['error_details'])}"
                       f" ({len(report.errors)})</summary><ul>")
            for err in report.errors:
                out.append(f"<li>{e(str(err)[:800])}</li>")
            out.append("</ul></details>")
        out.append(_markdown_to_html(_section(report, "summary")))
        out.append(f"<details><summary>{e(LABELS['research'])}</summary>")
        out.append(_markdown_to_html(_section(report, "research")))
        out.append("</details>")
        out.append(f"<details><summary>{e(LABELS['rationales'])}</summary>")
        out.append(_markdown_to_html(_section(report, "forecast_rationales")))
        out.append("</details></div>")

    out.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

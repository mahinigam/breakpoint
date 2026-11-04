"""Build the Tableau Public workbook from the aggregate export.

The repository ships 12 aggregate CSVs in ``dashboards/data``.  This script
turns the files that belong in the reviewer dashboard into a real, portable
Tableau workbook (``dashboards/breakpoint_policy.twb``) and a packaged
``.twbx`` with the CSVs embedded under ``Data/``.

It intentionally uses only Tableau's documented workbook XML.  The generated
workbook is checked by ``tests/test_tableau_workbook.py`` and CI rebuilds it
byte-for-byte; the structural rules come from Tableau's published 2026.2
workbook schema and the patterns used by hand-authored workbooks.

Usage:
    python scripts/build_tableau_workbook.py
    python scripts/build_tableau_workbook.py --check
"""

from __future__ import annotations

# Workbook XML structure follows Tableau's published schema and the
# MIT-licensed vizwright chart patterns (github.com/blakefeiza/vizwright).
import argparse
import csv
import io
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dashboards" / "data"
TWB = ROOT / "dashboards" / "breakpoint_policy.twb"
TWBX = ROOT / "dashboards" / "breakpoint_policy.twbx"

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "kpi": {
        "caption": "KPI — scenario economics",
        "file": "kpi.csv",
        "columns": [
            ("scenario_label", "string", "dimension", "nominal"),
            ("profit_uplift_pct", "real", "measure", "quantitative"),
        ],
    },
    "action_mix": {
        "caption": "Action mix",
        "file": "action_mix.csv",
        "columns": [
            ("scenario_label", "string", "dimension", "nominal"),
            ("action", "string", "dimension", "nominal"),
            ("share_pct", "real", "measure", "quantitative"),
        ],
    },
    "threshold": {
        "caption": "Threshold curve",
        "file": "threshold_curve.csv",
        "columns": [
            ("scenario_label", "string", "dimension", "nominal"),
            ("probability_cutoff", "real", "dimension", "quantitative"),
            ("profit_millions", "real", "measure", "quantitative"),
        ],
    },
    "power": {
        "caption": "Champion / challenger power",
        "file": "power.csv",
        "columns": [
            ("n_accounts_per_arm", "integer", "dimension", "quantitative"),
            ("power_pct", "real", "measure", "quantitative"),
        ],
    },
    "calibration": {
        "caption": "Calibration",
        "file": "calibration.csv",
        "columns": [
            ("method", "string", "dimension", "nominal"),
            ("mean_predicted_pct", "real", "measure", "quantitative"),
            ("observed_pct", "real", "measure", "quantitative"),
        ],
    },
}

# ---------------------------------------------------------------------------
# XML primitives
# ---------------------------------------------------------------------------


def _esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&apos;")
        .replace('"', "&quot;")
    )


def _column_decl(name: str, datatype: str, role: str, type_: str, indent: str = "      ") -> str:
    return (
        f"{indent}<column datatype='{datatype}' name='[{name}]' "
        f"role='{role}' type='{type_}' />"
    )


def _instance_name(name: str, kind: str) -> str:
    if kind == "none" and name in ("scenario_label", "action", "method"):
        return f"none:{name}:nk"
    if kind == "none":
        return f"none:{name}:qk"
    return f"{kind}:{name}:qk"


def _instance_xml(ds: str, name: str, kind: str) -> str:
    derivation = {
        "none": "None",
        "sum": "Sum",
        "avg": "Avg",
        "cnt": "Count",
        "ctd": "CountD",
    }[kind]
    type_ = "nominal" if kind == "none" and name in ("scenario_label", "action", "method") else "quantitative"
    return (
        f"            <column-instance column='[{name}]' derivation='{derivation}' "
        f"name='[{_instance_name(name, kind)}]' pivot='key' type='{type_}' />"
    )


def _ref(ds: str, instance: str) -> str:
    return f"[federated.{ds}].[{instance}]"


def _datasource(ds_name: str, spec: dict[str, Any], directory: str) -> str:
    columns = "\n".join(
        f"            <column datatype='{datatype}' name='{name}' ordinal='{index}' />"
        for index, (name, datatype, _, _) in enumerate(spec["columns"])
    )
    declarations = "\n".join(
        _column_decl(name, datatype, role, type_)
        for name, datatype, role, type_ in spec["columns"]
    )
    return f"""    <datasource caption='{_esc(spec["caption"])}' inline='true' name='federated.{ds_name}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption='{_esc(spec["caption"])}' name='textscan.{ds_name}'>
            <connection class='textscan' directory='{_esc(directory)}' filename='{_esc(spec["file"])}' workgroup-auth-mode='as-is' />
          </named-connection>
        </named-connections>
        <relation connection='textscan.{ds_name}' name='{_esc(spec["file"])}' table='[{Path(spec["file"]).stem}#csv]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{columns}
          </columns>
        </relation>
      </connection>
{declarations}
    </datasource>"""


def _worksheet(
    *,
    name: str,
    title: str,
    ds_name: str,
    fields: list[tuple[str, str]],
    rows: str,
    cols: str,
    mark: str,
    encodings: list[tuple[str, str]] | None = None,
    filters: list[str] | None = None,
) -> str:
    """Render one worksheet.

    ``fields`` is a list of ``(field_name, instance_kind)`` pairs.  Every raw
    field and every instance is declared in the dependencies block, which is
    what Tableau requires for a workbook to load outside Desktop.
    """
    raw_decls: dict[str, str] = {}
    instance_decls: list[str] = []
    instance_by_name: dict[str, str] = {}
    for field_name, kind in fields:
        raw_decls[field_name] = next(
            _column_decl(n, dt, role, typ)
            for n, dt, role, typ in DATASETS[ds_name]["columns"]
            if n == field_name
        )
        instance = _instance_name(field_name, kind)
        instance_decls.append(_instance_xml(ds_name, field_name, kind))
        instance_by_name[field_name] = instance

    enc_xml = ""
    if encodings:
        enc_xml = "\n".join(
            f"              <{tag} column='{_ref(ds_name, instance_by_name[field])}' />"
            for tag, field in encodings
        )
        enc_xml = f"\n            <encodings>\n{enc_xml}\n            </encodings>"

    filter_xml = ""
    if filters:
        filter_xml = "\n" + "\n".join(f"          {item}" for item in filters)

    return f"""    <worksheet name='{_esc(name)}'>
      <layout-options>
        <title>
          <formatted-text>
            <run bold='true' fontcolor='#222222' fontname='Tableau Medium' fontsize='12'>{_esc(title)}</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='{_esc(DATASETS[ds_name]["caption"])}' name='federated.{ds_name}' />
          </datasources>
          <datasource-dependencies datasource='federated.{ds_name}'>
{chr(10).join(raw_decls.values())}
{chr(10).join(instance_decls)}
          </datasource-dependencies>{filter_xml}
          <aggregation value='true' />
        </view>
        <style>
          <style-rule element='gridline'>
            <format attr='stroke-size' scope='rows' value='0' />
            <format attr='stroke-size' scope='cols' value='0' />
          </style-rule>
          <style-rule element='table-div'>
            <format attr='stroke-size' scope='rows' value='0' />
            <format attr='line-visibility' scope='rows' value='off' />
            <format attr='stroke-size' scope='cols' value='0' />
            <format attr='line-visibility' scope='cols' value='off' />
          </style-rule>
          <style-rule element='mark'>
            <format attr='mark-labels-show' value='true' />
            <format attr='mark-labels-cull' value='true' />
          </style-rule>
          <style-rule element='worksheet'>
            <format attr='display-field-labels' scope='cols' value='false' />
            <format attr='display-field-labels' scope='rows' value='false' />
            <format attr='font-family' value='Tableau Book' />
            <format attr='color' value='#222222' />
          </style-rule>
        </style>
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='{mark}' />{enc_xml}
          </pane>
        </panes>
        <rows>{rows}</rows>
        <cols>{cols}</cols>
      </table>
    </worksheet>"""


CARDS = """      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='2147483647'>
            <card type='columns' />
          </strip>
          <strip size='2147483647'>
            <card type='rows' />
          </strip>
          <strip size='31'>
            <card type='title' />
          </strip>
        </edge>
      </cards>"""


def _windows(sheet_names: list[str], dashboard_name: str) -> str:
    lines = ["  <windows source-height='30'>"]
    for sheet in sheet_names:
        lines.append(f"    <window class='worksheet' hidden='true' name='{_esc(sheet)}'>")
        lines.append(CARDS)
        lines.append("    </window>")
    lines.append(f"    <window class='dashboard' maximized='true' name='{_esc(dashboard_name)}'>")
    lines.append("      <viewpoints>")
    for sheet in sheet_names:
        lines.append(f"        <viewpoint name='{_esc(sheet)}'>")
        lines.append("          <zoom type='entire-view' />")
        lines.append("        </viewpoint>")
    lines.append("      </viewpoints>")
    lines.append("      <active id='-1' />")
    lines.append("      <device-preview selected='Desktop' />")
    lines.append("    </window>")
    lines.append("  </windows>")
    return "\n".join(lines)


def _dashboard(policy_text: str) -> str:
    def zone(zid: int, x: int, y: int, w: int, h: int, name: str | None = None, text: str | None = None) -> str:
        attrs = f"h='{h}' w='{w}' x='{x}' y='{y}' id='{zid}'"
        if text is not None:
            return f"""          <zone {attrs} type-v2='text' forceUpdate='true'>
            <formatted-text>
              <run>{_esc(text)}</run>
            </formatted-text>
            <zone-style>
              <format attr='border-style' value='none' />
              <format attr='padding' value='12' />
            </zone-style>
          </zone>"""
        return f"""          <zone {attrs} name='{_esc(name or "")}' show-title='true'>
            <zone-style>
              <format attr='border-style' value='none' />
              <format attr='margin' value='8' />
              <format attr='padding' value='12' />
              <format attr='background-color' value='#ffffff' />
            </zone-style>
          </zone>"""

    title = (
        "BREAKPOINT  /  GROW · HOLD · RESTRICT\n"
        "Calibrated credit risk, priced as a decision"
    )
    footnote = (
        "Source: reports/metrics.json, reports/policy.json, reports/bootstrap.json. "
        "Population-weighted OOF evidence; economics are labelled assumptions."
    )
    return f"""    <dashboard name='Breakpoint — Policy Brief'>
      <style />
      <size maxheight='1000' maxwidth='1500' minheight='1000' minwidth='1500' sizing-mode='fixed' />
      <zones>
        <zone h='100000' w='100000' x='0' y='0' id='1' param='vert' type-v2='layout-flow'>
{zone(2, 2000, 0, 96000, 7000, text=title)}
{zone(3, 2000, 7000, 96000, 8000, text=policy_text)}
{zone(4, 2000, 16000, 31000, 34000, name='Profit uplift')}
{zone(5, 34000, 16000, 31000, 34000, name='Action mix')}
{zone(6, 66000, 16000, 31000, 34000, name='Power')}
{zone(7, 2000, 51000, 62000, 45000, name='Threshold curve')}
{zone(8, 65000, 51000, 32000, 45000, name='Calibration')}
{zone(9, 2000, 97000, 96000, 3000, text=footnote)}
          <zone-style>
            <format attr='border-style' value='none' />
            <format attr='padding' value='8' />
            <format attr='background-color' value='#f4f1eb' />
          </zone-style>
        </zone>
      </zones>
    </dashboard>"""


def _policy_text() -> str:
    path = DATA_DIR / "policy_card.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return (
        f"Grow ≤ {float(row['grow_cutoff']):.6f}  ·  Hold ≤ {float(row['hold_upper_cutoff']):.6f}  ·  "
        f"Restrict > {float(row['hold_upper_cutoff']):.6f}     "
        f"Worst-case regret {float(row['action_worst_case_regret_pct']):.2f}% vs "
        f"{float(row['binary_worst_case_regret_pct']):.2f}% for binary minimax"
    )


def _workbook(directory: str, policy_text: str) -> str:
    datasources = "\n".join(
        _datasource(ds_name, spec, directory) for ds_name, spec in DATASETS.items()
    )

    sheets = [
        _worksheet(
            name="Profit uplift",
            title="Profit uplift vs binary minimax",
            ds_name="kpi",
            fields=[("scenario_label", "none"), ("profit_uplift_pct", "sum")],
            rows=_ref("kpi", "none:scenario_label:nk"),
            cols=_ref("kpi", "sum:profit_uplift_pct:qk"),
            mark="Bar",
            encodings=[("text", "profit_uplift_pct")],
        ),
        _worksheet(
            name="Action mix",
            title="Grow / Hold / Restrict mix",
            ds_name="action_mix",
            fields=[
                ("scenario_label", "none"),
                ("action", "none"),
                ("share_pct", "sum"),
            ],
            rows=_ref("action_mix", "none:scenario_label:nk"),
            cols=_ref("action_mix", "sum:share_pct:qk"),
            mark="Bar",
            encodings=[("color", "action")],
        ),
        _worksheet(
            name="Threshold curve",
            title="Profit by cutoff (cutoff ≤ 25%)",
            ds_name="threshold",
            fields=[
                ("scenario_label", "none"),
                ("probability_cutoff", "none"),
                ("profit_millions", "sum"),
            ],
            rows=_ref("threshold", "sum:profit_millions:qk"),
            cols=_ref("threshold", "none:probability_cutoff:qk"),
            mark="Line",
            encodings=[("color", "scenario_label")],
            filters=[
                "<filter class='quantitative' column='[federated.threshold].[none:probability_cutoff:qk]' included-values='in-range'>"
                "<min>0.0</min><max>0.25</max></filter>",
                "<slices><column>[federated.threshold].[none:probability_cutoff:qk]</column></slices>",
            ],
        ),
        _worksheet(
            name="Power",
            title="Two-arm power curve",
            ds_name="power",
            fields=[("n_accounts_per_arm", "none"), ("power_pct", "sum")],
            rows=_ref("power", "sum:power_pct:qk"),
            cols=_ref("power", "none:n_accounts_per_arm:qk"),
            mark="Line",
        ),
        _worksheet(
            name="Calibration",
            title="Reliability: predicted vs observed",
            ds_name="calibration",
            fields=[
                ("method", "none"),
                ("mean_predicted_pct", "sum"),
                ("observed_pct", "sum"),
            ],
            rows=_ref("calibration", "sum:observed_pct:qk"),
            cols=_ref("calibration", "sum:mean_predicted_pct:qk"),
            mark="Circle",
            encodings=[("color", "method")],
        ),
    ]

    return f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook original-version='18.1' source-build='2026.2.0 (20262.26.0410.0924)' source-platform='mac' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences />
  <datasources>
{datasources}
  </datasources>
  <worksheets>
{chr(10).join(sheets)}
  </worksheets>
  <dashboards>
{_dashboard(policy_text)}
  </dashboards>
{_windows(["Profit uplift", "Action mix", "Threshold curve", "Power", "Calibration"], "Breakpoint — Policy Brief")}
</workbook>
"""


FIXED_ZIP_DATE = (2026, 11, 3, 0, 0, 0)


def _package_bytes(policy_text: str) -> bytes:
    """Return a byte-deterministic .twbx archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        twb_info = zipfile.ZipInfo("breakpoint_policy.twb", date_time=FIXED_ZIP_DATE)
        twb_info.compress_type = zipfile.ZIP_DEFLATED
        twb_info.external_attr = 0o644 << 16
        archive.writestr(twb_info, _workbook("Data", policy_text).encode("utf-8"))
        for spec in DATASETS.values():
            info = zipfile.ZipInfo(f"Data/{spec['file']}", date_time=FIXED_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (DATA_DIR / spec["file"]).read_bytes())
    return buffer.getvalue()


def render() -> tuple[str, bytes]:
    policy_text = _policy_text()
    return _workbook("data", policy_text), _package_bytes(policy_text)


def build(*, check: bool = False) -> int:
    twb_text, twbx_bytes = render()
    if check:
        stale = []
        if not TWB.exists() or TWB.read_text(encoding="utf-8") != twb_text:
            stale.append(str(TWB.relative_to(ROOT)))
        if not TWBX.exists() or TWBX.read_bytes() != twbx_bytes:
            stale.append(str(TWBX.relative_to(ROOT)))
        if stale:
            print("stale generated workbook(s): " + ", ".join(stale))
            print("run: python scripts/build_tableau_workbook.py")
            return 1
        print("generated workbooks are current")
        return 0

    TWB.write_text(twb_text, encoding="utf-8")
    TWBX.write_bytes(twbx_bytes)
    print(f"wrote {TWB.relative_to(ROOT)} ({TWB.stat().st_size:,} bytes)")
    print(f"wrote {TWBX.relative_to(ROOT)} ({TWBX.stat().st_size:,} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed workbook is stale")
    args = parser.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())

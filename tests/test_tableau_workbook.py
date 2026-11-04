import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARDS = ROOT / "dashboards"
DATA = DASHBOARDS / "data"
TWB = DASHBOARDS / "breakpoint_policy.twb"
TWBX = DASHBOARDS / "breakpoint_policy.twbx"

EXPECTED_DATASETS = {
    "federated.kpi": "kpi.csv",
    "federated.action_mix": "action_mix.csv",
    "federated.threshold": "threshold_curve.csv",
    "federated.power": "power.csv",
    "federated.calibration": "calibration.csv",
}
EXPECTED_SHEETS = {"Profit uplift", "Action mix", "Threshold curve", "Power", "Calibration"}


def _root(path):
    return ET.parse(path).getroot()


def test_source_workbook_references_existing_aggregate_files():
    root = _root(TWB)
    datasources = root.findall("./datasources/datasource")
    assert {ds.get("name") for ds in datasources} == set(EXPECTED_DATASETS)

    for datasource in datasources:
        name = datasource.get("name")
        connection = datasource.find(
            "./connection/named-connections/named-connection/connection"
        )
        assert connection is not None
        assert connection.get("class") == "textscan"
        assert connection.get("filename") == EXPECTED_DATASETS[name]
        assert (DATA / connection.get("filename")).is_file()


def test_workbook_has_dashboard_sheets_and_window_state():
    root = _root(TWB)
    sheets = {sheet.get("name") for sheet in root.findall("./worksheets/worksheet")}
    assert sheets == EXPECTED_SHEETS

    dashboard = root.find("./dashboards/dashboard")
    assert dashboard is not None
    assert dashboard.get("name") == "Breakpoint — Policy Brief"

    zone_names = {
        zone.get("name")
        for zone in dashboard.findall("./zones/zone/zone")
        if zone.get("name")
    }
    assert zone_names == EXPECTED_SHEETS

    windows = root.findall("./windows/window")
    assert {window.get("name") for window in windows} == EXPECTED_SHEETS | {
        "Breakpoint — Policy Brief"
    }
    assert any(window.get("class") == "dashboard" for window in windows)
    viewpoints = root.findall("./windows/window[@class='dashboard']/viewpoints/viewpoint")
    assert {point.get("name") for point in viewpoints} == EXPECTED_SHEETS


def test_packaged_workbook_is_portable_and_contains_every_file():
    with zipfile.ZipFile(TWBX) as archive:
        names = set(archive.namelist())
        assert "breakpoint_policy.twb" in names
        assert {f"Data/{name}" for name in EXPECTED_DATASETS.values()} <= names
        packaged = ET.fromstring(archive.read("breakpoint_policy.twb"))

    for datasource in packaged.findall("./datasources/datasource"):
        connection = datasource.find(
            "./connection/named-connections/named-connection/connection"
        )
        assert connection is not None
        assert connection.get("directory") == "Data"
        assert connection.get("filename") in EXPECTED_DATASETS.values()


def test_committed_workbook_is_reproducible():
    result = subprocess.run(
        [sys.executable, "scripts/build_tableau_workbook.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

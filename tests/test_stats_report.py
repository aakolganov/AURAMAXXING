"""Tests for stats.report (figure + JSON output) and the `python -m stats` CLI."""
import json

from base.amorphous_structure import AmorphousStruc_factory
from stats import write_report, write_pooled_report, analyze_structure

GEN_PNGS = {"coordination.png", "homo_element_distance.png", "element_O_distance.png", "tau4.png"}
SAT_PNGS = GEN_PNGS | {"oh_distance.png", "oh_per_element.png"}


def _struct():
    pos = [[10, 10, 10], [11.6, 10, 10], [8.4, 10, 10], [10, 11.6, 10], [10, 8.4, 10]]
    return AmorphousStruc_factory(symbols=["Si", "O", "O", "O", "O"], positions=pos,
                                  cell=[20.0, 20.0, 20.0], pbc=True, seed=0)


def test_write_report_creates_figures_and_json(tmp_path):
    write_report(_struct(), tmp_path, is_saturation=False, label="t")
    files = {p.name for p in tmp_path.iterdir()}
    assert GEN_PNGS <= files
    assert "stats.json" in files
    data = json.loads((tmp_path / "stats.json").read_text())
    assert data["label"] == "t" and "summary" in data and "metrics" in data


def test_saturation_report_adds_oh_plots(tmp_path):
    s = AmorphousStruc_factory(symbols=["Si", "O", "H"],
                               positions=[[5, 5, 5], [6.6, 5, 5], [7.5, 5, 5]],
                               cell=[20.0, 20.0, 20.0], pbc=True, seed=0)
    write_report(s, tmp_path, is_saturation=True)
    files = {p.name for p in tmp_path.iterdir()}
    assert SAT_PNGS <= files


def test_pooled_report(tmp_path):
    metrics = [analyze_structure(_struct()), analyze_structure(_struct())]
    write_pooled_report(metrics, tmp_path, is_saturation=False)
    files = {p.name for p in tmp_path.iterdir()}
    assert GEN_PNGS <= files
    data = json.loads((tmp_path / "stats_pooled.json").read_text())
    assert data["summary"]["n_structures"] == 2


def test_cli_on_structure_file(tmp_path):
    from stats.__main__ import main
    # write a structure to a POSCAR, then analyse it via the CLI
    s = _struct()
    poscar = tmp_path / "POSCAR"
    s.atoms.write(str(poscar), format="vasp")
    out = tmp_path / "stats_out"
    rc = main([str(poscar), "--out", str(out)])
    assert rc == 0
    assert (out / "stats.json").exists()
    assert GEN_PNGS <= {p.name for p in out.iterdir()}

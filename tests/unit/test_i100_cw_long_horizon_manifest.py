from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_long_horizon_manifest_has_exact_resume_and_horizons(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts/build_i100_cw_long_horizon_manifest.py"
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"status": "complete_no_update", "tau": 2.0, "beta_advce": .1,
                               "margin_coefficient": .2, "margin_floor": .1, "margin_cap": .3}))
    paths = []
    for name in ("d1.yaml", "d2.yaml", "m1.json", "m2.json"):
        path = tmp_path / name; path.write_text("{}\n"); paths.append(path)
    out = tmp_path / "manifest.json"
    cmd = [sys.executable, str(script), "--output", str(out), "--source-sha", "a" * 40,
           "--root", str(tmp_path / "root"), "--repo", str(tmp_path / "repo"), "--calibration", str(cal),
           "--dev1-config", str(paths[0]), "--dev2-config", str(paths[1]), "--dev1-mask", str(paths[2]), "--dev2-mask", str(paths[3])]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(out.read_text())
    assert len(manifest["jobs"]) == 12
    trains = [j for j in manifest["jobs"] if j["job_id"].startswith("train-")]
    assert len(trains) == 6
    for job in trains:
        argv = job["command"]
        assert argv[argv.index("--resume-epoch") + 1] == "114"
        assert argv[argv.index("--epochs") + 1] == "200"
        assert argv[argv.index("--horizon-epochs") + 1:argv.index("--run-namespace")] == ["129", "149", "169", "189", "199"]
    assert {j["arm"] if "arm" in j else j["scientific_identity"]["arm"] for j in trains} == {"I100_CONTROL", "CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM"}
    assert all(j["host"] == "ferret" for j in manifest["jobs"])

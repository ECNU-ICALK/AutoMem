import json

import automem.cli as cli


def test_space_command_prints_machine_readable_manifest(capsys):
    assert cli.main(["space"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["counts"]["cartesian_total"] == 3720
    assert output["counts"]["encode_subsets"] == 31
    assert output["counts"]["compatible_total"] == 2573
    assert output["space"]["manage"] == [
        "lightweight",
        "json_full",
        "tool_manager",
        "graph_consolidate",
    ]


def test_offline_smoke_runs_full_lifecycle(capsys):
    assert cli.main(["smoke"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "ok"
    assert set(output["checks"].values()) == {"ok"}


def test_smoke_command_returns_nonzero_on_failure(monkeypatch, capsys):
    def fail():
        raise RuntimeError("forced failure")

    monkeypatch.setattr(cli, "run_offline_smoke", fail)

    assert cli.main(["smoke"]) == 1
    assert "forced failure" in capsys.readouterr().err

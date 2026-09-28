import json

from tiny_transformer.experiments.demo import main, run_demo


def test_demo_is_deterministic_and_reports_its_training_evidence() -> None:
    first = run_demo(prompt="attention uses", max_new_tokens=3, steps=2)
    second = run_demo(prompt="attention uses", max_new_tokens=3, steps=2)

    assert first == second
    assert first["prompt"] == "attention uses"
    assert len(first["generated_tokens"]) == 3
    assert first["training"]["steps"] == 2
    assert first["training"]["final_loss"] < first["training"]["initial_loss"]
    assert first["limitations"]


def test_demo_cli_prints_machine_readable_json(capsys) -> None:
    assert main(["--prompt", "attention uses", "--tokens", "2", "--steps", "1"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["prompt_tokens"] == ["attention", "uses"]
    assert len(report["generated_tokens"]) == 2

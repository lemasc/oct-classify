from pathlib import Path

from oct_classify.data.config import load_dataset_specs


def test_load_dataset_specs_defaults_to_enabled_and_reads_disabled_sources(tmp_path: Path) -> None:
    config = tmp_path / "datasets.toml"
    config.write_text(
        """
[[datasets]]
name = "enabled"
source = "duke"
root = "datasets/enabled"
available_labels = ["normal"]

[[datasets]]
name = "disabled"
source = "octid"
root = "datasets/disabled"
available_labels = ["normal", "amd"]
enabled = false
""".lstrip(),
        encoding="utf-8",
    )

    enabled, disabled = load_dataset_specs(config)

    assert enabled.enabled is True
    assert disabled.enabled is False

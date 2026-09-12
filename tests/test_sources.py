from pathlib import Path

from oct_classify.data.models import DatasetSpec
from oct_classify.data.sources.duke import DukeSource
from oct_classify.data.sources.kermany import KermanySource
from oct_classify.data.sources.octdl import OctdlSource
from oct_classify.data.sources.octid import OctidSource
from oct_classify.data.sources.paima import PaimaSource
from oct_classify.data.taxonomy import ALL_LABELS, UnifiedLabel


def _spec(name: str, root: Path) -> DatasetSpec:
    return DatasetSpec(name=name, source=name, root=root, available_labels=ALL_LABELS)


def test_duke_uses_class_folder_as_group_id(tmp_path: Path) -> None:
    folder = tmp_path / "AMD1"
    folder.mkdir()
    (folder / "scan.png").touch()

    records = list(DukeSource().records(_spec("duke", tmp_path)))

    assert records[0].label is UnifiedLabel.AMD
    assert records[0].group_id == "AMD1"


def test_kermany_merges_cnv_and_drusen_into_amd(tmp_path: Path) -> None:
    for label, group_id in (("CNV", "123"), ("DRUSEN", "456")):
        folder = tmp_path / "train" / label
        folder.mkdir(parents=True)
        (folder / f"{label}-{group_id}-1.jpeg").touch()

    records = list(KermanySource().records(_spec("kermany", tmp_path)))

    assert {record.label for record in records} == {UnifiedLabel.AMD}
    assert {record.supplied_split for record in records} == {"train"}
    assert {record.group_id for record in records} == {"123", "456"}


def test_octdl_joins_metadata_to_image_and_patient(tmp_path: Path) -> None:
    image_dir = tmp_path / "NO"
    image_dir.mkdir()
    (image_dir / "normal_1.jpg").touch()
    (tmp_path / "labels.csv").write_text(
        "file_name,disease,patient_id\nnormal_1,NO,patient-1\n", encoding="utf-8"
    )

    records = list(OctdlSource().records(_spec("octdl", tmp_path)))

    assert records[0].label is UnifiedLabel.NORMAL
    assert records[0].group_id == "patient-1"


def test_octid_includes_only_normal_and_amrd_images(tmp_path: Path) -> None:
    for label in ("AMRD", "NORMAL", "CSR"):
        folder = tmp_path / label
        folder.mkdir()
        (folder / "scan.jpeg").touch()

    records = list(OctidSource().records(_spec("octid", tmp_path)))

    assert {(record.raw_label, record.label) for record in records} == {
        ("AMRD", UnifiedLabel.AMD),
        ("NORMAL", UnifiedLabel.NORMAL),
    }
    assert all(record.group_id is None for record in records)


def test_paima_uses_metadata_labels_and_patient_groups(tmp_path: Path) -> None:
    image_dir = tmp_path / "CNV" / "1"
    image_dir.mkdir(parents=True)
    (image_dir / "scan.tif").touch()
    (tmp_path / "data_information.csv").write_text(
        "Patient ID,Class,Eye,B-scan,Label,Directory\npatient-1,CNV,OD,0,CNV,CNV/1/SCAN.TIF\n",
        encoding="utf-8",
    )

    records = list(PaimaSource().records(_spec("paima", tmp_path)))

    assert records[0].label is UnifiedLabel.AMD
    assert records[0].group_id == "CNV/patient-1"

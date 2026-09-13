import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_root_cli_import_does_not_import_torch() -> None:
    _run("import sys; import oct_classify.cli; assert 'torch' not in sys.modules")


def test_data_help_does_not_import_torch() -> None:
    _run(
        "import sys\n"
        "from oct_classify.cli import main\n"
        "try:\n"
        "    main(['data', '--help'])\n"
        "except SystemExit as error:\n"
        "    assert error.code == 0\n"
        "assert 'torch' not in sys.modules\n"
    )


def test_cli_help_lists_command_families() -> None:
    result = _run(
        "from oct_classify.cli import main\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit as error:\n"
        "    assert error.code == 0\n"
    )

    assert "data" in result.stdout
    assert "train" in result.stdout

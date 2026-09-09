from unittest.mock import patch

import run


def test_run_pipeline_calls_train_then_export() -> None:
    calls = []

    def fake_train(*args, **kwargs):
        calls.append(("train", kwargs))

    def fake_export(*args, **kwargs):
        calls.append(("export", kwargs))

    with (
        patch("run.train", side_effect=fake_train),
        patch("run.export", side_effect=fake_export),
    ):
        with patch(
            "sys.argv",
            [
                "run.py",
                "--model-name",
                "xor",
                "--model-version",
                "v2",
            ],
        ):
            run.main()

    assert calls == [
        (
            "train",
            {
                "model_name": "xor",
                "model_version": "v2",
            },
        ),
        (
            "export",
            {
                "model_name": "xor",
                "model_version": "v2",
            },
        ),
    ]
import json

import numpy as np
import pytest
import yaml

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.pipelines import interpolate_local_nersi as pipeline
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from tests.fixtures.c3_volume_run_artifacts import prepare_c3_volume_run_artifacts


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("fit_alignment", [False, True])
def test_local_pipeline_real_poc_loader_target_invariance_and_records(
    tmp_path, monkeypatch, accumulation, fit_alignment
):
    predictions, metrics = [], []
    for number, offset in enumerate((0.0, 20.0)):
        artifacts = prepare_c3_volume_run_artifacts(
            tmp_path / f"inputs{number}",
            dataset_id="seg_c3_na",
            missing_fraction=0.8,
            time_sample_count=8,
            receiver_y_count=16 if fit_alignment else 8,
            target_offset=offset,
        )
        selection = artifacts.volume_metadata["selection"]
        dimensions = C3BenchmarkDimensions(
            time_range=tuple(selection["time"]),
            sail_line_numbers=(selection["source_line"][0], selection["source_line"][1] - 1),
            shape=tuple(artifacts.volume_metadata["shape"]),
        )
        monkeypatch.setattr(
            pipeline,
            "load_c3_random80_poc_inputs",
            lambda dimensions=dimensions, **kwargs: load_c3_random80_poc_inputs(
                **kwargs, dimensions=dimensions
            ),
        )
        config = load_resolved_config("studies/study_040_c3_nersi_target_tuning/local_lines8.yaml")
        config["benchmark_case"]["id"] = "synthetic_case"
        config["benchmark_volume"].update(id="synthetic_volume", selection=selection)
        config["model"].update(
            fourier_components=2,
            encoder_width=8,
            latent_channels=2,
            decoder_channels=[2, 2, 1],
            kernel_size=1,
        )
        config["training"].update(max_steps=2, report_interval=1, profiles_per_step=1, device="cpu")
        config["training"]["gradient_accumulation_steps"] = accumulation
        config["local_models"]["source_line_width"] = 1
        if fit_alignment:
            config["local_models"]["alignment_search"] = {
                "candidates": [1.0, 2.0, 3.0],
                "distances": list(range(1, 16)),
            }
        config["trace_rms_idw"]["radius"] = 8
        path = tmp_path / f"config{number}.yaml"
        path.write_text(yaml.safe_dump(config))
        output = tmp_path / f"run{number}"
        paths = {
            key + "_dir": getattr(artifacts, key)
            for key in ("interim", "processed", "mask", "case", "volume")
        }
        with pytest.raises(ValueError, match="frozen full input lock"):
            pipeline.interpolate_local_nersi_run(
                config_path=path,
                output_dir=output,
                input_paths=paths,
                expected_inputs_lock={"wrong": "inputs"},
            )
        assert not output.exists()
        metrics.append(
            pipeline.interpolate_local_nersi_run(
                config_path=path,
                output_dir=output,
                input_paths=paths,
                reporter=lambda _: None,
            )
        )
        predictions.append(np.load(output / "prediction.npy"))
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["status"] == "success"
        if fit_alignment:
            assert all(f["fit_domain"] == "observed_only" for f in metadata["time_alignment_fits"])
            assert len(metadata["time_alignment_fits"]) == dimensions.shape[1]
        assert metadata["coverage"]["complete"]
        assert metadata["checkpoint_restoration_max_abs_error"] == 0
        assert metadata["compute"]["optimizer_updates"] == 2 * dimensions.shape[1]
        assert all(r["gradient_accumulation_steps"] == accumulation for r in metadata["blocks"])
        with pytest.raises(FileExistsError):
            pipeline.interpolate_local_nersi_run(
                config_path=path, output_dir=output, input_paths=paths
            )
    np.testing.assert_array_equal(*predictions)
    assert metrics[0]["normalized_reference"] != metrics[1]["normalized_reference"]
    assert metrics[0]["physical"] != metrics[1]["physical"]

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
import torch
import yaml

from tutorial.case.functors.case_relic_wrappers import CaseMEGANModel
from tutorial.case.run_case import (
    _resolve_resume_context,
    build_simulator,
    initialize_trainable_map,
    load_saved_trainables_from_run,
    load_yaml,
    register_case_parameters,
)
from tutorial.case.utils.netcdf_loader import CaseNetCDFLoader, LoaderConfig
from tutorial.case.utils.optimization_loop import OptimizationLoopEngine


@pytest.fixture(autouse=True)
def _drop_case_parameter_sources(fresh_global_parameter_manager):
    """register_case_parameters adds the case parameter file (it overrides the
    calibration bounds, tags and requires_grad of biosphere.bvoc.megan.* and
    biosphere.photosynthesis.* keys) as a source of the process-wide manager and
    reloads it; every test works on a fresh manager and the manager that was
    global before is put back afterwards, so later test modules read the
    packaged database only."""
    yield


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@contextmanager
def _workspace_tempdir(prefix: str):
    temp_root = _repo_root() / "tmp_test_artifacts"
    temp_root.mkdir(parents=True, exist_ok=True)
    work_dir = temp_root / f"{prefix}{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield work_dir
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _load_optimization_yaml() -> dict:
    repo_root = _repo_root()
    yaml_path = repo_root / "tutorial" / "case" / "config" / "optimization.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    assert isinstance(payload, dict)
    optimization = payload.get("optimization")
    assert isinstance(optimization, dict)
    return optimization


def test_case_megan_model_exposes_isoprene_trainables_as_parameters() -> None:
    repo_root = _repo_root()
    case_yaml = repo_root / "tutorial" / "case" / "config" / "case_parameters.yaml"
    register_case_parameters(case_yaml)

    model = CaseMEGANModel(name="test_megan_bvoc")
    trainables = model.initialize_case_trainables()

    assert "biosphere.bvoc.default.isoprene" in trainables
    assert "biosphere.bvoc.megan.alpha_light" in trainables
    assert "biosphere.bvoc.megan.cl_light" in trainables
    assert "biosphere.bvoc.megan.eopt_isoprene" in trainables
    assert "biosphere.bvoc.megan.topt" in trainables
    assert "biosphere.bvoc.megan.ct1_isoprene" in trainables
    assert "biosphere.bvoc.megan.ct2" in trainables
    # The per-species MEGAN keys are retired: T_opt and C_T2 are class independent
    # (Guenther et al. 2012 Eq. 9 and Sect. 2.2).
    assert "biosphere.bvoc.megan.topt_isoprene" not in trainables
    assert "biosphere.bvoc.megan.ct2_isoprene" not in trainables

    assert isinstance(model.case_isoprene_ef, torch.nn.Parameter)
    assert isinstance(model.case_alpha_light, torch.nn.Parameter)
    assert isinstance(model.case_cl_light, torch.nn.Parameter)
    assert isinstance(model.case_eopt_isoprene, torch.nn.Parameter)
    assert isinstance(model.case_topt, torch.nn.Parameter)
    assert isinstance(model.case_ct1_isoprene, torch.nn.Parameter)
    assert isinstance(model.case_ct2, torch.nn.Parameter)

    assert trainables["biosphere.bvoc.megan.alpha_light"] is model.case_alpha_light
    assert trainables["biosphere.bvoc.megan.cl_light"] is model.case_cl_light
    assert trainables["biosphere.bvoc.megan.eopt_isoprene"] is model.case_eopt_isoprene
    assert trainables["biosphere.bvoc.megan.topt"] is model.case_topt
    assert trainables["biosphere.bvoc.megan.ct1_isoprene"] is model.case_ct1_isoprene
    assert trainables["biosphere.bvoc.megan.ct2"] is model.case_ct2
    # The promoted tensors are the ones the MEGAN temperature activity reads.
    assert model.topt is model.case_topt
    assert model.ct2 is model.case_ct2

    # Initial values are the database values: E_opt = C_eo = 2 for isoprene
    # (Guenther et al. 2012 Table 4), T_opt = 313 K (Eq. 9 at T240 = 297 K),
    # C_T2 = 230 kJ mol-1 (Sect. 2.2).
    assert float(model.case_eopt_isoprene.detach()) == pytest.approx(2.0)
    assert float(model.case_topt.detach()) == pytest.approx(313.0)
    assert float(model.case_ct2.detach()) == pytest.approx(230000.0)


def test_case_megan_temperature_response_retains_gradient_chain() -> None:
    repo_root = _repo_root()
    case_yaml = repo_root / "tutorial" / "case" / "config" / "case_parameters.yaml"
    register_case_parameters(case_yaml)

    model = CaseMEGANModel(name="test_megan_temp_grad")
    model.initialize_case_trainables()
    temperature = torch.tensor([303.15, 307.15, 311.15], dtype=torch.float32)

    factor = model._get_temperature_factor("isoprene", temperature)
    factor.mean().backward()

    for parameter in (
        model.case_eopt_isoprene,
        model.case_topt,
        model.case_ct1_isoprene,
        model.case_ct2,
    ):
        assert parameter is not None
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(torch.abs(parameter.grad).sum().item()) > 0.0


def test_optimization_yaml_uses_only_supported_high_signal_trainables() -> None:
    optimization = _load_optimization_yaml()

    phase2 = optimization["phase2"]
    phase3 = optimization["phase3"]
    phase4 = optimization["phase4"]

    assert phase2["trainable_keys"] == [
        "tutorial.case.radiation.par_effective_scale",
        "biosphere.photosynthesis.vcmax25",
        "biosphere.photosynthesis.jmax25",
        "biosphere.photosynthesis.rd25",
    ]

    assert phase3["trainable_keys"] == [
        "biosphere.bvoc.default.isoprene",
        "biosphere.bvoc.megan.alpha_light",
        "biosphere.bvoc.megan.cl_light",
        "biosphere.bvoc.megan.eopt_isoprene",
        "biosphere.bvoc.megan.topt",
        "biosphere.bvoc.megan.ct1_isoprene",
        "biosphere.bvoc.megan.ct2",
    ]

    assert phase4["trainable_keys"] == [
        "tutorial.case.radiation.par_effective_scale",
        "biosphere.photosynthesis.vcmax25",
        "biosphere.photosynthesis.jmax25",
        "biosphere.photosynthesis.rd25",
        "biosphere.bvoc.default.isoprene",
        "biosphere.bvoc.megan.alpha_light",
        "biosphere.bvoc.megan.cl_light",
        "biosphere.bvoc.megan.eopt_isoprene",
        "biosphere.bvoc.megan.topt",
        "biosphere.bvoc.megan.ct1_isoprene",
        "biosphere.bvoc.megan.ct2",
    ]

    forbidden_zero_gradient = {
        "biosphere.bvoc.megan.beta_temperature",
        "hydrosphere.soil_moisture.wilting_point",
    }

    for phase in (phase2, phase3, phase4):
        assert forbidden_zero_gradient.isdisjoint(set(phase["trainable_keys"]))


def test_optimization_yaml_disables_de_and_uses_gradient_only_fine_tuning() -> None:
    optimization = _load_optimization_yaml()
    de_cfg = optimization["de_preoptimization"]

    assert de_cfg["enabled"] is False
    assert de_cfg["parallel"] is False
    assert de_cfg["objective_steps"] == 48
    assert de_cfg["population_size_factor"] == 4
    assert de_cfg["phases"]["phase1"]["enabled"] is False
    assert de_cfg["phases"]["phase2"]["enabled"] is False
    assert de_cfg["phases"]["phase3"]["enabled"] is True
    assert de_cfg["phases"]["phase3"]["max_iterations"] == 6
    assert de_cfg["phases"]["phase4"]["enabled"] is False


def test_case_parameter_bounds_and_daytime_loss_configuration_match_no_de_finetune_plan() -> None:
    repo_root = _repo_root()
    case_yaml = repo_root / "tutorial" / "case" / "config" / "case_parameters.yaml"
    payload = yaml.safe_load(case_yaml.read_text(encoding="utf-8-sig"))
    assert isinstance(payload, dict)
    parameters = payload["parameters"]

    assert parameters["biosphere.bvoc.megan.alpha_light"]["bounds"] == [0.0003, 0.05]
    assert parameters["biosphere.bvoc.megan.alpha_light"]["value"] == 0.0027
    assert parameters["biosphere.bvoc.megan.cl_light"]["bounds"] == [0.5, 3.0]
    assert parameters["biosphere.bvoc.megan.cl_light"]["value"] == 1.066
    # Class-independent T_opt and C_T2 (Guenther et al. 2012 Eq. 9, Sect. 2.2)
    # replace the retired per-species keys; the isoprene E_opt starts at the
    # Table 4 value C_eo = 2 with the database dtype.
    for retired in ("topt_isoprene", "ct2_isoprene", "topt_iso"):
        assert f"biosphere.bvoc.megan.{retired}" not in parameters
    assert parameters["biosphere.bvoc.megan.eopt_isoprene"]["value"] == 2.0
    assert parameters["biosphere.bvoc.megan.eopt_isoprene"]["dtype"] == "float32"
    assert parameters["biosphere.bvoc.megan.topt"]["value"] == 313.0
    assert parameters["biosphere.bvoc.megan.topt"]["units"] == "K"
    assert parameters["biosphere.bvoc.megan.ct2"]["value"] == 230000.0
    assert parameters["biosphere.bvoc.megan.ct2"]["units"] == "J mol-1"
    assert parameters["hydrosphere.soil_moisture.saturated_hydraulic_conductivity"]["bounds"] == [
        1.0e-08,
        1.0e-04,
    ]
    assert parameters["hydrosphere.soil_moisture.saturated_hydraulic_conductivity"]["value"] == 1.0e-06

    optimization = _load_optimization_yaml()
    for phase_name in ("phase1", "phase3"):
        for key in optimization[phase_name]["trainable_keys"]:
            entry = parameters[key]
            lower, upper = entry["bounds"]
            assert lower < upper
            assert lower <= entry["value"] <= upper
            assert entry["requires_grad"] is True
            assert "trainable" in entry["tags"]

    # The MEGAN slope beta_temperature had no reader and is removed from the database,
    # so the case no longer declares a fixed override for it.
    assert "biosphere.bvoc.megan.beta_temperature" not in parameters
    assert all(
        "biosphere.bvoc.megan.beta_temperature" not in phase["trainable_keys"]
        for phase in optimization.values()
        if isinstance(phase, dict) and "trainable_keys" in phase
    )

    phase3 = optimization["phase3"]
    phase4 = optimization["phase4"]

    assert optimization["phase1"]["lr"] == 0.0008
    assert optimization["phase1"]["epochs"] == 4
    assert phase3["lr"] == 0.00045
    assert phase3["epochs"] == 18
    assert phase4["lr"] == 2.5e-05
    assert phase4["epochs"] == 6
    assert phase3["daytime_loss"] == {
        "enabled": True,
        "par_key": "ppfd",
        "par_threshold": 10.0,
        "target_state_keys": ["isoprene_emission"],
        "positive_weight": 2.0,
        "underprediction_penalty_weight": 1.0,
    }
    assert phase4["daytime_loss"] == {
        "enabled": True,
        "par_key": "ppfd",
        "par_threshold": 10.0,
        "target_state_keys": ["isoprene_emission"],
        "positive_weight": 1.5,
        "underprediction_penalty_weight": 0.75,
    }


def test_de_preoptimization_updates_parameters_within_bounds() -> None:
    class FakeStateManager:
        def __init__(self) -> None:
            self.state_vars = {}

        def reset(self) -> None:
            self.state_vars = {}

        def set_state(self, updates, validate: bool = False) -> None:
            del validate
            self.state_vars.update(updates)

        def get_state(self, clone: bool = False):
            return dict(self.state_vars) if clone else self.state_vars

    class FakeSimulator:
        def __init__(self, trainable_param: torch.nn.Parameter) -> None:
            self.state_manager = FakeStateManager()
            self.trainable_param = trainable_param
            self.current_time = 0.0

        def step(self, dt_seconds: float) -> None:
            x_value = float(self.trainable_param.detach().item())
            prediction = (x_value - 2.5) ** 2
            self.state_manager.set_state(
                {
                    "isoprene_emission": torch.tensor([prediction], dtype=torch.float32),
                },
                validate=False,
            )
            self.current_time += float(dt_seconds)

    class FakeLoader:
        total_steps = 4

        def get_forcing_step(self, step: int):
            return {"forcing_step": torch.tensor([float(step)], dtype=torch.float32)}

        def get_obs_step(self, step: int, obs_key: str):
            del step
            assert obs_key == "isoprene_emission_obs"
            return torch.tensor([0.0], dtype=torch.float32)

        def get_obs_weight_multiplier(self, obs_key: str) -> float:
            del obs_key
            return 1.0

        def get_obs_quality_label(self, obs_key: str) -> str:
            del obs_key
            return "real_observation"

    with _workspace_tempdir(prefix="relic_de_warmstart_") as temp_dir:
        param = torch.nn.Parameter(torch.tensor(4.5, dtype=torch.float32))
        simulator = FakeSimulator(param)
        loader = FakeLoader()
        engine = OptimizationLoopEngine(
            simulator=simulator,
            loader=loader,
            dt_seconds=3600.0,
            gradient_clip_norm=1.0,
            show_progress=False,
            checkpoint_dir=temp_dir,
        )

        def reset_state() -> None:
            simulator.current_time = 0.0
            simulator.state_manager.reset()

        result = engine.run_de_preoptimization(
            phase_name="phase3",
            phase_cfg={
                "target_state_key": "isoprene_emission",
                "obs_key": "isoprene_emission_obs",
                "trainable_keys": ["biosphere.bvoc.default.isoprene"],
            },
            de_cfg={
                "enabled": True,
                "objective_steps": 4,
                "max_iterations": 8,
                "population_size_factor": 6,
                "strategy": "best1bin",
                "mutation_factor": (0.5, 1.0),
                "crossover_probability": 0.8,
                "tolerance": 1.0e-6,
                "seed": 42,
                "parallel": False,
            },
            trainable_params={"biosphere.bvoc.default.isoprene": param},
            param_bounds={"biosphere.bvoc.default.isoprene": (0.0, 5.0)},
            reset_hook=reset_state,
        )

        assert result is not None
        assert result.objective_value < result.initial_objective_value
        assert 0.0 <= result.optimized_parameters["biosphere.bvoc.default.isoprene"] <= 5.0
        assert abs(result.optimized_parameters["biosphere.bvoc.default.isoprene"] - 2.5) < 0.5
        assert result.checkpoint_path is not None
        assert result.checkpoint_path.exists()


def test_de_preoptimization_propagates_simulation_errors() -> None:
    """RELIC declares no divergence error type that a DE objective could map to an
    infinite objective, so an error raised by a candidate simulation propagates out of
    run_de_preoptimization instead of being scored as +inf and hidden."""

    class FailingStateManager:
        def reset(self) -> None:
            return None

        def set_state(self, updates, validate: bool = False) -> None:
            del updates, validate

        def get_state(self, clone: bool = False):
            del clone
            return {"isoprene_emission": torch.tensor([0.0], dtype=torch.float32)}

    class FailingSimulator:
        def __init__(self) -> None:
            self.state_manager = FailingStateManager()
            self.current_time = 0.0

        def step(self, dt_seconds: float) -> None:
            del dt_seconds
            raise RuntimeError("candidate simulation failed")

    class FakeLoader:
        total_steps = 2

        def get_forcing_step(self, step: int):
            return {"forcing_step": torch.tensor([float(step)], dtype=torch.float32)}

        def get_obs_step(self, step: int, obs_key: str):
            del step, obs_key
            return torch.tensor([0.0], dtype=torch.float32)

        def get_obs_weight_multiplier(self, obs_key: str) -> float:
            del obs_key
            return 1.0

        def get_obs_quality_label(self, obs_key: str) -> str:
            del obs_key
            return "real_observation"

    with _workspace_tempdir(prefix="relic_de_error_") as temp_dir:
        param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        engine = OptimizationLoopEngine(
            simulator=FailingSimulator(),
            loader=FakeLoader(),
            dt_seconds=3600.0,
            gradient_clip_norm=1.0,
            show_progress=False,
            checkpoint_dir=temp_dir,
        )
        with pytest.raises(RuntimeError, match="candidate simulation failed"):
            engine.run_de_preoptimization(
                phase_name="phase3",
                phase_cfg={
                    "target_state_key": "isoprene_emission",
                    "obs_key": "isoprene_emission_obs",
                    "trainable_keys": ["biosphere.bvoc.default.isoprene"],
                },
                de_cfg={
                    "enabled": True,
                    "objective_steps": 2,
                    "max_iterations": 2,
                    "population_size_factor": 4,
                    "strategy": "best1bin",
                    "mutation_factor": (0.5, 1.0),
                    "crossover_probability": 0.8,
                    "tolerance": 1.0e-6,
                    "seed": 7,
                    "parallel": False,
                },
                trainable_params={"biosphere.bvoc.default.isoprene": param},
                param_bounds={"biosphere.bvoc.default.isoprene": (0.0, 5.0)},
                reset_hook=lambda: None,
            )


def test_initialize_trainable_map_excludes_unsupported_zero_gradient_keys() -> None:
    repo_root = _repo_root()
    case_root = repo_root / "tutorial" / "case"
    config_dir = case_root / "config"
    data_dir = case_root / "data" / "processed"

    register_case_parameters(config_dir / "case_parameters.yaml")
    simulation_cfg = load_yaml(config_dir / "simulation.yaml")
    simulation_cfg["simulation"]["device"] = "cpu"
    simulation_cfg["simulation"]["require_gpu"] = False
    simulator = build_simulator(simulation_cfg)
    loader = CaseNetCDFLoader(
        LoaderConfig(
            case_root=case_root,
            forcing_path=data_dir / "site_forcing.nc",
            obs_path=data_dir / "real_obs.nc",
            case_parameter_path=config_dir / "case_parameters.yaml",
            device=torch.device("cpu"),
        )
    )
    trainable = initialize_trainable_map(simulator, loader)

    optimization = _load_optimization_yaml()
    for key in optimization["phase1"]["trainable_keys"]:
        assert key in trainable
    assert "biosphere.photosynthesis.rd25" in trainable
    assert "biosphere.bvoc.megan.alpha_light" in trainable
    assert "biosphere.bvoc.megan.cl_light" in trainable
    assert "biosphere.bvoc.megan.eopt_isoprene" in trainable
    assert "biosphere.bvoc.megan.topt" in trainable
    assert "biosphere.bvoc.megan.ct1_isoprene" in trainable
    assert "biosphere.bvoc.megan.ct2" in trainable
    assert "biosphere.bvoc.megan.topt_isoprene" not in trainable
    assert "biosphere.bvoc.megan.ct2_isoprene" not in trainable
    assert "biosphere.bvoc.megan.beta_temperature" not in trainable


def test_load_saved_trainables_from_run_reads_phase_checkpoint() -> None:
    with _workspace_tempdir(prefix="relic_saved_trainables_") as temp_dir:
        output_root = temp_dir / "runs"
        checkpoint_dir = output_root / "sample_run" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint_phase3_epoch0002.pt"
        torch.save(
            {
                "trainable_params": {
                    "biosphere.bvoc.default.isoprene": torch.tensor(18.5),
                    "biosphere.bvoc.megan.topt": torch.tensor(309.15),
                }
            },
            checkpoint_path,
        )

        loaded = load_saved_trainables_from_run(
            output_root=output_root,
            run_id="sample_run",
            phase_name="phase3",
        )

        assert loaded["biosphere.bvoc.default.isoprene"] == 18.5
        assert loaded["biosphere.bvoc.megan.topt"] == 309.15


def test_run_phase_writes_final_checkpoint_with_completed_phase() -> None:
    class FakeStateManager:
        def __init__(self) -> None:
            self.state_vars = {}

        def reset(self) -> None:
            self.state_vars = {}

        def set_state(self, updates, validate: bool = False) -> None:
            del validate
            self.state_vars.update(updates)

        def get_state(self, clone: bool = False):
            return dict(self.state_vars) if clone else self.state_vars

    class FakeSimulator:
        def __init__(self, trainable_param: torch.nn.Parameter) -> None:
            self.state_manager = FakeStateManager()
            self.trainable_param = trainable_param
            self.current_time = 0.0

        def step(self, dt_seconds: float) -> None:
            prediction = (self.trainable_param - 1.5) ** 2
            self.state_manager.set_state(
                {
                    "isoprene_emission": prediction.reshape(1),
                },
                validate=False,
            )
            self.current_time += float(dt_seconds)

    class FakeLoader:
        total_steps = 2

        def get_forcing_step(self, step: int):
            return {"forcing_step": torch.tensor([float(step)], dtype=torch.float32)}

        def get_obs_step(self, step: int, obs_key: str):
            del step
            assert obs_key == "isoprene_emission_obs"
            return torch.tensor([0.0], dtype=torch.float32)

        def get_obs_weight_multiplier(self, obs_key: str) -> float:
            del obs_key
            return 1.0

        def get_obs_quality_label(self, obs_key: str) -> str:
            del obs_key
            return "real_observation"

    with _workspace_tempdir(prefix="relic_phase_ckpt_") as temp_dir:
        param = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        simulator = FakeSimulator(param)
        loader = FakeLoader()
        engine = OptimizationLoopEngine(
            simulator=simulator,
            loader=loader,
            dt_seconds=3600.0,
            gradient_clip_norm=1.0,
            show_progress=False,
            checkpoint_dir=temp_dir,
        )

        def reset_state() -> None:
            simulator.current_time = 0.0
            simulator.state_manager.reset()

        outputs = engine.run_phase(
            phase_name="phase2",
            phase_cfg={
                "lr": 0.05,
                "epochs": 1,
                "window_hours": 1,
                "target_state_key": "isoprene_emission",
                "obs_key": "isoprene_emission_obs",
                "trainable_keys": ["biosphere.bvoc.default.isoprene"],
            },
            trainable_params={"biosphere.bvoc.default.isoprene": param},
            param_bounds={"biosphere.bvoc.default.isoprene": (0.0, 5.0)},
            reset_hook=reset_state,
            completed_phases=["phase1"],
        )

        assert outputs.loss_rows
        checkpoint_path = temp_dir / "checkpoint_phase2_epoch0000.pt"
        assert checkpoint_path.exists()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert payload["completed_phases"] == ["phase1", "phase2"]


def test_run_phase_resume_requires_compatible_optimizer_state() -> None:
    class FakeStateManager:
        def __init__(self) -> None:
            self.state_vars = {}

        def reset(self) -> None:
            self.state_vars = {}

        def set_state(self, updates, validate: bool = False) -> None:
            del validate
            self.state_vars.update(updates)

        def get_state(self, clone: bool = False):
            return dict(self.state_vars) if clone else self.state_vars

    class FakeSimulator:
        def __init__(self, trainable_param: torch.nn.Parameter) -> None:
            self.state_manager = FakeStateManager()
            self.trainable_param = trainable_param
            self.current_time = 0.0

        def step(self, dt_seconds: float) -> None:
            prediction = (self.trainable_param - 0.5) ** 2
            self.state_manager.set_state(
                {
                    "isoprene_emission": prediction.reshape(1),
                },
                validate=False,
            )
            self.current_time += float(dt_seconds)

    class FakeLoader:
        total_steps = 1

        def get_forcing_step(self, step: int):
            return {"forcing_step": torch.tensor([float(step)], dtype=torch.float32)}

        def get_obs_step(self, step: int, obs_key: str):
            del step
            assert obs_key == "isoprene_emission_obs"
            return torch.tensor([0.0], dtype=torch.float32)

        def get_obs_weight_multiplier(self, obs_key: str) -> float:
            del obs_key
            return 1.0

        def get_obs_quality_label(self, obs_key: str) -> str:
            del obs_key
            return "real_observation"

    with _workspace_tempdir(prefix="relic_resume_optimizer_") as temp_dir:
        param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        simulator = FakeSimulator(param)
        loader = FakeLoader()
        engine = OptimizationLoopEngine(
            simulator=simulator,
            loader=loader,
            dt_seconds=3600.0,
            gradient_clip_norm=1.0,
            show_progress=False,
            checkpoint_dir=temp_dir,
        )

        def reset_state() -> None:
            simulator.current_time = 0.0
            simulator.state_manager.reset()

        try:
            engine.run_phase(
                phase_name="phase3",
                phase_cfg={
                    "lr": 0.05,
                    "epochs": 1,
                    "window_hours": 1,
                    "target_state_key": "isoprene_emission",
                    "obs_key": "isoprene_emission_obs",
                    "trainable_keys": ["biosphere.bvoc.default.isoprene"],
                },
                trainable_params={"biosphere.bvoc.default.isoprene": param},
                param_bounds={"biosphere.bvoc.default.isoprene": (0.0, 5.0)},
                reset_hook=reset_state,
                resume_optimizer_state={"invalid": True},
            )
        except RuntimeError as exc:
            assert "Failed to restore optimizer state" in str(exc)
        else:
            raise AssertionError("Expected strict resume to reject incompatible optimizer state.")


def test_resolve_resume_context_reads_phase_checkpoint_and_de_warmstart() -> None:
    with _workspace_tempdir(prefix="relic_resume_context_") as temp_dir:
        output_root = temp_dir / "runs"
        checkpoint_dir = output_root / "sample_run" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "phase_name": "phase2",
                "epoch": 3,
                "completed_phases": ["phase1", "phase2"],
                "trainable_params": {
                    "biosphere.bvoc.default.isoprene": torch.tensor(19.0),
                },
                "optimizer_state_dict": {"state": {}, "param_groups": []},
                "epoch_loss": 1.25,
            },
            checkpoint_dir / "checkpoint_phase2_epoch0003.pt",
        )
        torch.save(
            {
                "kind": "de_warm_start",
                "phase_name": "phase3",
                "optimized_parameters": {
                    "biosphere.bvoc.default.isoprene": 17.5,
                },
                "objective_value": 0.75,
                "initial_objective_value": 2.5,
                "iteration_count": 4,
                "success": True,
                "termination_reason": "manual_test",
                "convergence_history": [2.5, 1.5, 0.75],
            },
            checkpoint_dir / "de_phase3_warmstart.pt",
        )

        context = _resolve_resume_context(
            output_root=output_root,
            run_id="sample_run",
        )

        assert context.run_id == "sample_run"
        assert context.resume_phase == "phase2"
        assert context.resume_epoch == 4
        assert context.completed_phases == ["phase1", "phase2"]
        assert context.resume_epoch_loss == 1.25
        assert "phase3" in context.de_warmstart_payloads
        assert (
            context.de_warmstart_payloads["phase3"]["optimized_parameters"]["biosphere.bvoc.default.isoprene"]
            == 17.5
        )

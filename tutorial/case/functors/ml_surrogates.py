"""ML surrogate functors for the Tapajos tutorial case."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.models.pytorch_surrogate import ModelLoadError, PyTorchSurrogate


def _merge_surrogate_config(
    default_config: Dict[str, Any],
    override_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(default_config)
    if not isinstance(override_config, dict):
        return merged
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


class _BiologicalProcessSurrogateBridge(PyTorchSurrogate):
    """Bridge-backed PyTorch surrogate base for biological tutorial processes."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        metadata: Optional[FunctorMetadata] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        resolved_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        resolved_metadata = metadata
        if resolved_metadata is None:
            resolved_metadata = FunctorMetadata(
                name=name,
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                description=f"Bridge-backed biological surrogate: {name}",
            )
        else:
            resolved_metadata.category = ProcessCategory.BIOLOGICAL_PROCESS

        kwargs.setdefault("process_name", name)

        super().__init__(
            name=name,
            config=config,
            metadata=resolved_metadata,
            device=resolved_device,
            **kwargs,
        )


class _TorchGradientSafeSurrogate(_BiologicalProcessSurrogateBridge):
    """PyTorch-only surrogate I/O path that preserves autograd graphs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._input_mean: Optional[torch.Tensor] = None
        self._input_scale: Optional[torch.Tensor] = None
        self._output_mean: Optional[torch.Tensor] = None
        self._output_scale: Optional[torch.Tensor] = None
        super().__init__(*args, **kwargs)

    def _coerce_scalar_value(self, value: object) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().item())
            return float(value.detach().mean().item())
        return float(value)

    def _fit_normalizers_from_pairs(
        self,
        states: list[StateDict],
        fluxes: list[FluxDict],
    ) -> None:
        if not states or not fluxes:
            return
        input_rows = [
            [
                self._coerce_scalar_value(
                    state[str(self.input_mapping.get(var_name, var_name))]
                )
                for var_name in self.input_variables
            ]
            for state in states
        ]
        output_rows = [
            [self._coerce_scalar_value(flux[var_name]) for var_name in self.output_variables]
            for flux in fluxes
        ]
        dtype = self._model_dtype()
        input_tensor = torch.as_tensor(input_rows, dtype=dtype, device=self.device)
        output_tensor = torch.as_tensor(output_rows, dtype=dtype, device=self.device)
        epsilon = torch.as_tensor(1.0e-6, dtype=dtype, device=self.device)
        self._input_mean = input_tensor.mean(dim=0, keepdim=True)
        self._input_scale = torch.clamp(input_tensor.std(dim=0, unbiased=False, keepdim=True), min=epsilon)
        self._output_mean = output_tensor.mean(dim=0, keepdim=True)
        self._output_scale = torch.clamp(output_tensor.std(dim=0, unbiased=False, keepdim=True), min=epsilon)

    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._input_mean is None or self._input_scale is None:
            return inputs
        return (inputs - self._input_mean) / self._input_scale

    def _normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self._output_mean is None or self._output_scale is None:
            return targets
        return (targets - self._output_mean) / self._output_scale

    def _denormalize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        if self._output_mean is None or self._output_scale is None:
            return outputs
        return outputs * self._output_scale + self._output_mean

    def _move_batch_to_device_dtype(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_inputs, batch_targets = super()._move_batch_to_device_dtype(batch_inputs, batch_targets)
        batch_inputs = self._normalize_inputs(batch_inputs)
        batch_targets = self._normalize_targets(batch_targets)
        return batch_inputs, batch_targets

    def train(
        self,
        training_data: Any,
        validation_data: Optional[Any] = None,
        epochs: Optional[int] = None,
        verbose: bool = True,
        validate_units: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(training_data, tuple) and len(training_data) == 2:
            states, fluxes = training_data
            if isinstance(states, list) and isinstance(fluxes, list):
                self._fit_normalizers_from_pairs(states, fluxes)
        return super().train(
            training_data=training_data,
            validation_data=validation_data,
            epochs=epochs,
            verbose=verbose,
            validate_units=validate_units,
        )

    def _require_checkpoint_features(self, checkpoint_path: Path) -> None:
        """Accept a checkpoint only for the feature definition it was trained on.

        save_model records the ordered input_variables and output_variables with
        the weights. Weights trained on another input vector have the same shape
        whenever the feature count agrees, so a name mismatch is an error here,
        never a silent load.
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ModelLoadError(
                f"{self.name}: checkpoint {checkpoint_path} is not a mapping with recorded "
                "input features; retrain it through the case training path "
                "(simulation.ml_surrogate.train_from_physics) after removing the file."
            )
        saved_inputs = checkpoint.get("input_variables")
        saved_outputs = checkpoint.get("output_variables")
        current_inputs = list(self.input_variables)
        current_outputs = list(self.output_variables)
        if saved_inputs != current_inputs or saved_outputs != current_outputs:
            raise ModelLoadError(
                f"{self.name}: checkpoint {checkpoint_path} records input features "
                f"{saved_inputs} and outputs {saved_outputs}, but the surrogate declares "
                f"input features {current_inputs} and outputs {current_outputs}; retrain it "
                "through the case training path (simulation.ml_surrogate.train_from_physics) "
                "after removing the stale file."
            )

    def _restore_normalizers_from_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            return
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        dtype = self._model_dtype()
        for attr_name, checkpoint_key in (
            ("_input_mean", "input_mean"),
            ("_input_scale", "input_scale"),
            ("_output_mean", "output_mean"),
            ("_output_scale", "output_scale"),
        ):
            tensor = checkpoint.get(checkpoint_key)
            if isinstance(tensor, torch.Tensor):
                setattr(self, attr_name, tensor.to(device=self.device, dtype=dtype))
            else:
                setattr(self, attr_name, None)

    def _load_model(self) -> None:
        # A missing file builds an untrained network (library rule); an existing
        # file must match the declared features and load, or the error propagates.
        checkpoint_path = Path(self.model_path)
        if checkpoint_path.exists():
            self._require_checkpoint_features(checkpoint_path)
        super()._load_model()
        self._restore_normalizers_from_checkpoint(checkpoint_path)

    def save_model(self, path: Optional[Path] = None) -> None:
        super().save_model(path=path)
        save_path = Path(path or self.model_path)
        checkpoint = torch.load(
            save_path,
            map_location="cpu",
            weights_only=True,
        )
        for attr_name, checkpoint_key in (
            ("_input_mean", "input_mean"),
            ("_input_scale", "input_scale"),
            ("_output_mean", "output_mean"),
            ("_output_scale", "output_scale"),
        ):
            tensor = getattr(self, attr_name, None)
            if isinstance(tensor, torch.Tensor):
                checkpoint[checkpoint_key] = tensor.detach().clone().cpu()
            else:
                checkpoint[checkpoint_key] = None
        torch.save(checkpoint, save_path)

    def _model_dtype(self) -> torch.dtype:
        if self.model is None:
            return torch.float32
        return next(self.model.parameters()).dtype

    def _predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(f"Model not created for surrogate {self.name}")
        if not torch.is_grad_enabled():
            outputs = super()._predict(self._normalize_inputs(input_tensor))
            return self._denormalize_outputs(outputs)
        input_tensor = input_tensor.to(device=self.device, dtype=self._model_dtype())
        input_tensor = self._normalize_inputs(input_tensor)
        if int(input_tensor.shape[0]) <= 1:
            with self._temporary_batchnorm_eval():
                outputs = self.model(input_tensor)
        else:
            outputs = self.model(input_tensor)
        return self._denormalize_outputs(outputs)

    def _coerce_feature_tensor(self, value: object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=self._model_dtype())
        else:
            tensor = torch.as_tensor(value, dtype=self._model_dtype(), device=self.device)
        if tensor.dim() == 0:
            return tensor.reshape(1)
        if tensor.dim() == 1:
            return tensor
        reduce_dims = tuple(range(1, tensor.dim()))
        return tensor.mean(dim=reduce_dims)

    def _prepare_input(self, state: StateDict) -> torch.Tensor:
        feature_tensors = []
        batch_size: Optional[int] = None
        for var_name in self.input_variables:
            state_var_name = str(self.input_mapping.get(var_name, var_name))
            if state_var_name not in state:
                raise KeyError(f"Required input variable not found: {state_var_name}")
            feature_tensor = self._coerce_feature_tensor(state[state_var_name])
            current_batch = int(feature_tensor.shape[0])
            if batch_size is None:
                batch_size = current_batch
            elif current_batch == 1 and batch_size > 1:
                feature_tensor = feature_tensor.expand(batch_size)
            elif batch_size == 1 and current_batch > 1:
                feature_tensors = [
                    existing.expand(current_batch) if int(existing.shape[0]) == 1 else existing
                    for existing in feature_tensors
                ]
                batch_size = current_batch
            elif current_batch != batch_size:
                raise ValueError(
                    f"Inconsistent surrogate batch size for {self.name}: "
                    f"{state_var_name} has {current_batch}, expected {batch_size}"
                )
            feature_tensors.append(feature_tensor)
        if not feature_tensors:
            raise ValueError(f"Surrogate {self.name} has no input features configured.")
        return torch.stack(feature_tensors, dim=1)

    def _format_output(self, predictions: torch.Tensor, dt: float) -> FluxDict:
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        if predictions.dim() != 2:
            raise ValueError(
                f"Invalid prediction tensor dimensions for {self.name}: {tuple(predictions.shape)}"
            )
        if int(predictions.shape[1]) != len(self.output_variables):
            raise ValueError(
                f"Model output dimension mismatch for {self.name}: "
                f"got {int(predictions.shape[1])}, expected {len(self.output_variables)}"
            )
        dt_tensor = torch.as_tensor(dt, dtype=predictions.dtype, device=predictions.device)
        outputs: FluxDict = {}
        for index, var_name in enumerate(self.output_variables):
            values = predictions[:, index]
            if "rate" in var_name.lower() or "flux" in var_name.lower():
                values = values * dt_tensor
            outputs[var_name] = values
        return outputs


class CaseSoilMoistureSurrogate(_TorchGradientSafeSurrogate):
    """Meteorology-driven soil-moisture surrogate for the tutorial case."""

    def __init__(
        self,
        name: str = "ml_soil_moisture",
        model_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_root = Path(model_dir or Path("models"))
        default_config: Dict[str, Any] = {
            "model_path": str(model_root / "sm_surrogate.pth"),
            "validate_model_path": False,
            "conservation_check": False,
            "uncertainty_quantification": False,
            "dependencies": [],
            "input_variables": [
                "air_temperature",
                "ppfd",
                "vapor_pressure_deficit",
                "precipitation_flux",
                "net_radiation",
                "wind_speed",
                "leaf_area_index",
            ],
            "output_variables": ["soil_moisture_ML"],
            "output_units": {
                "soil_moisture_ML": "m3 m-3",
            },
            "architecture": {
                "type": "resnet",
                "hidden_dim": 64,
                "n_blocks": 3,
                "activation": "relu",
                "dropout_rate": 0.05,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 8.0e-4,
                "batch_size": 32,
                "epochs": 220,
                "loss_function": "huber",
                "early_stopping_patience": 20,
                "validation_split": 0.2,
                "shuffle_training": False,
            },
            "uncertainty": {
                "method": "dropout_mc",
                "mc_samples": 30,
            },
        }
        merged_config = _merge_surrogate_config(default_config, config)
        declared_inputs = [
            str(merged_config.get("input_mapping", {}).get(var_name, var_name))
            for var_name in merged_config["input_variables"]
        ]
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="ML surrogate for case soil moisture under uncertain forcing.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "theta_ML = NN(T_air, PPFD, VPD, P, Rn, U, LAI)"
            ),
            references=[
                "Ball et al. (1987). Progress in Photosynthesis Research.",
                "Razavi et al. (2012). Water Resources Research 48, W07401.",
                "Rodriguez-Iturbe and Porporato (2004). Ecohydrology of Water-Controlled Ecosystems.",
            ],
            parameters={
                "tutorial.case.ml_surrogate.soil_moisture.output_min": "Lower clipping bound for soil_moisture_ML [m3 m-3]",
                "tutorial.case.ml_surrogate.soil_moisture.output_max": "Upper clipping bound for soil_moisture_ML [m3 m-3]",
                "tutorial.case.ml_surrogate.soil_moisture.model_path": "Serialized PyTorch checkpoint path for soil moisture surrogate",
            },
            input_variables=declared_inputs,
            output_variables=["soil_moisture_ML"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=merged_config, metadata=metadata)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Razavi et al. (2012), surrogate emulation for hydrologic responses.
        # - Rodriguez-Iturbe and Porporato (2004), physically admissible soil-moisture range.
        # Mathematical expression:
        # - theta_ML_raw = NN(x_forcing)
        # - theta_ML = clip(theta_ML_raw, theta_min, theta_max)
        # Parameter meanings:
        # - theta_min: lower soil moisture bound [m3 m-3]
        # - theta_max: upper soil moisture bound [m3 m-3]
        outputs = super().forward(state, time, dt, **kwargs)
        soil_moisture_ml = outputs.get("soil_moisture_ML")
        if soil_moisture_ml is None:
            return outputs
        lower = self.get_parameter_value(
            "output_min",
            namespace="tutorial.case.ml_surrogate.soil_moisture",
            required=True,
        )
        upper = self.get_parameter_value(
            "output_max",
            namespace="tutorial.case.ml_surrogate.soil_moisture",
            required=True,
        )
        lower_tensor = torch.as_tensor(
            float(lower),
            dtype=soil_moisture_ml.dtype,
            device=soil_moisture_ml.device,
        )
        upper_tensor = torch.as_tensor(
            float(upper),
            dtype=soil_moisture_ml.dtype,
            device=soil_moisture_ml.device,
        )
        outputs["soil_moisture_ML"] = torch.clamp(
            soil_moisture_ml,
            min=lower_tensor,
            max=upper_tensor,
        )
        return outputs


class CaseWaterStressSurrogate(_TorchGradientSafeSurrogate):
    """ML surrogate for nonlinear soil-moisture stress in the tutorial case."""

    def __init__(
        self,
        name: str = "ml_water_stress",
        model_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_root = Path(model_dir or Path("models"))
        default_input_variables = [
            "soil_moisture_proxy",
            "air_temperature",
            "vapor_pressure_deficit",
            "leaf_area_index",
        ]
        default_config: Dict[str, Any] = {
            "model_path": str(model_root / "water_stress_surrogate.pth"),
            "validate_model_path": False,
            "conservation_check": False,
            "uncertainty_quantification": False,
            "dependencies": ["ml_soil_moisture"],
            "input_variables": default_input_variables,
            "input_mapping": {
                "soil_moisture_proxy": "soil_moisture_ML",
            },
            "input_units": {
                "soil_moisture_proxy": "m3 m-3",
            },
            "output_variables": ["water_stress_factor_ML"],
            "output_units": {
                "water_stress_factor_ML": "1",
            },
            "architecture": {
                "type": "mlp",
                "hidden_dims": [32, 16],
                "activation": "relu",
                "dropout_rate": 0.05,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 5.0e-4,
                "batch_size": 32,
                "epochs": 150,
                "loss_function": "huber",
                "early_stopping_patience": 15,
                "validation_split": 0.2,
                "shuffle_training": False,
            },
            "uncertainty": {
                "method": "dropout_mc",
                "mc_samples": 20,
            },
        }
        merged_config = _merge_surrogate_config(default_config, config)
        declared_inputs = [
            str(merged_config.get("input_mapping", {}).get(var_name, var_name))
            for var_name in merged_config["input_variables"]
        ]
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="ML surrogate for nonlinear soil-water stress under uncertain hydraulic parameters.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "f_W_ML = NN(theta_ML, T_air, VPD, LAI)"
            ),
            references=[
                "Egea et al. (2011). Agricultural and Forest Meteorology 151, 1370-1384.",
                "Verhoef and Egea (2014). Hydrology and Earth System Sciences 18, 4883-4897.",
            ],
            parameters={
                "tutorial.case.ml_surrogate.water_stress.output_min": "Lower clipping bound for water_stress_factor_ML [-]",
                "tutorial.case.ml_surrogate.water_stress.output_max": "Upper clipping bound for water_stress_factor_ML [-]",
                "tutorial.case.ml_surrogate.water_stress.model_path": "Serialized PyTorch checkpoint path for water-stress surrogate",
            },
            input_variables=declared_inputs,
            output_variables=["water_stress_factor_ML"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=merged_config, metadata=metadata)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Egea et al. (2011), nonlinear water-stress effects on canopy conductance.
        # - Verhoef and Egea (2014), hydraulic parameter uncertainty under soil drying.
        # Mathematical expression:
        # - f_W_ML_raw = NN(theta_ML, T_air, VPD, LAI)
        # - f_W_ML = clip(f_W_ML_raw, f_W_min, f_W_max)
        # Parameter meanings:
        # - f_W_min/f_W_max: admissible water stress factor bounds [-]
        outputs = super().forward(state, time, dt, **kwargs)
        water_stress_ml = outputs.get("water_stress_factor_ML")
        if water_stress_ml is None:
            return outputs
        lower = self.get_parameter_value(
            "output_min",
            namespace="tutorial.case.ml_surrogate.water_stress",
            required=True,
        )
        upper = self.get_parameter_value(
            "output_max",
            namespace="tutorial.case.ml_surrogate.water_stress",
            required=True,
        )
        lower_tensor = torch.as_tensor(
            float(lower),
            dtype=water_stress_ml.dtype,
            device=water_stress_ml.device,
        )
        upper_tensor = torch.as_tensor(
            float(upper),
            dtype=water_stress_ml.dtype,
            device=water_stress_ml.device,
        )
        outputs["water_stress_factor_ML"] = torch.clamp(
            water_stress_ml,
            min=lower_tensor,
            max=upper_tensor,
        )
        return outputs


class CaseStomatalSurrogate(_TorchGradientSafeSurrogate):
    """ML surrogate for lagged stomatal conductance in the tutorial case."""

    def __init__(
        self,
        name: str = "ml_stomatal",
        model_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_root = Path(model_dir or Path("models"))
        default_input_variables = [
            "air_temperature",
            "vapor_pressure_deficit",
            "ppfd",
            "co2_concentration",
            "net_photosynthesis",
            "soil_moisture_proxy",
        ]
        default_config: Dict[str, Any] = {
            "model_path": str(model_root / "gs_surrogate.pth"),
            "validate_model_path": False,
            "conservation_check": False,
            "uncertainty_quantification": False,
            "dependencies": ["farquhar_photosynthesis", "ml_soil_moisture"],
            "input_variables": default_input_variables,
            "input_mapping": {
                "soil_moisture_proxy": "soil_moisture_ML",
            },
            "input_units": {
                "co2_concentration": "ppm",
                "soil_moisture_proxy": "m3 m-3",
            },
            "output_variables": ["stomatal_conductance_ML"],
            "output_units": {
                "stomatal_conductance_ML": "mol m-2 s-1",
            },
            "architecture": {
                "type": "resnet",
                "hidden_dim": 64,
                "n_blocks": 2,
                "activation": "relu",
                "dropout_rate": 0.05,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 8.0e-4,
                "batch_size": 32,
                "epochs": 180,
                "loss_function": "huber",
                "early_stopping_patience": 18,
                "validation_split": 0.2,
                "shuffle_training": False,
            },
            "uncertainty": {
                "method": "dropout_mc",
                "mc_samples": 30,
            },
        }
        merged_config = _merge_surrogate_config(default_config, config)
        declared_inputs = [
            str(merged_config.get("input_mapping", {}).get(var_name, var_name))
            for var_name in merged_config["input_variables"]
        ]
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="ML surrogate for lagged stomatal conductance under case forcing uncertainty.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "g_s_ML = NN(T_air, VPD, PPFD, CO2, A_n, theta_proxy)"
            ),
            references=[
                "Ball et al. (1987). Progress in Photosynthesis Research.",
                "Leuning (1995). Plant, Cell and Environment 18, 339-355.",
                "Baldocchi (1994). Tree Physiology 14, 1069-1079.",
            ],
            parameters={
                "tutorial.case.ml_surrogate.stomatal.output_min": "Lower clipping bound for stomatal_conductance_ML [mol m-2 s-1]",
                "tutorial.case.ml_surrogate.stomatal.output_max": "Upper clipping bound for stomatal_conductance_ML [mol m-2 s-1]",
                "tutorial.case.ml_surrogate.stomatal.model_path": "Serialized PyTorch checkpoint path for stomatal surrogate",
            },
            input_variables=declared_inputs,
            output_variables=["stomatal_conductance_ML"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=merged_config, metadata=metadata)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Ball et al. (1987), assimilation-humidity control on stomatal conductance.
        # - Leuning (1995), VPD-sensitive stomatal response formulation.
        # Mathematical expression:
        # - g_s_ML_raw = NN(x_meteo, A_n, theta_proxy)
        # - g_s_ML = clip(g_s_ML_raw, g_min, g_max)
        # Parameter meanings:
        # - g_min: lower stomatal conductance bound [mol m-2 s-1]
        # - g_max: upper stomatal conductance bound [mol m-2 s-1]
        outputs = super().forward(state, time, dt, **kwargs)
        stomatal_ml = outputs.get("stomatal_conductance_ML")
        if stomatal_ml is None:
            return outputs
        lower = self.get_parameter_value(
            "output_min",
            namespace="tutorial.case.ml_surrogate.stomatal",
            required=True,
        )
        upper = self.get_parameter_value(
            "output_max",
            namespace="tutorial.case.ml_surrogate.stomatal",
            required=True,
        )
        lower_tensor = torch.as_tensor(
            float(lower),
            dtype=stomatal_ml.dtype,
            device=stomatal_ml.device,
        )
        upper_tensor = torch.as_tensor(
            float(upper),
            dtype=stomatal_ml.dtype,
            device=stomatal_ml.device,
        )
        outputs["stomatal_conductance_ML"] = torch.clamp(
            stomatal_ml,
            min=lower_tensor,
            max=upper_tensor,
        )
        return outputs


class CaseOzoneStressSurrogate(_TorchGradientSafeSurrogate):
    """ML surrogate for ozone damage factor in the tutorial case."""

    def __init__(
        self,
        name: str = "ml_ozone_stress",
        model_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_root = Path(model_dir or Path("models"))
        default_input_variables = [
            "stomatal_conductance_proxy",
            "air_temperature",
            "vapor_pressure_deficit",
            "o3_mixing_ratio",
        ]
        default_config: Dict[str, Any] = {
            "model_path": str(model_root / "ozone_stress_surrogate.pth"),
            "validate_model_path": False,
            "conservation_check": False,
            "uncertainty_quantification": False,
            "dependencies": ["ml_stomatal"],
            "input_variables": default_input_variables,
            "input_mapping": {
                "stomatal_conductance_proxy": "stomatal_conductance_ML",
            },
            "input_units": {
                "stomatal_conductance_proxy": "mol m-2 s-1",
                "o3_mixing_ratio": "mol mol-1",
            },
            "output_variables": ["ozone_stress_factor_ML"],
            "output_units": {
                "ozone_stress_factor_ML": "1",
            },
            "architecture": {
                "type": "mlp",
                "hidden_dims": [32, 16],
                "activation": "relu",
                "dropout_rate": 0.05,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 5.0e-4,
                "batch_size": 32,
                "epochs": 150,
                "loss_function": "huber",
                "early_stopping_patience": 15,
                "validation_split": 0.2,
                "shuffle_training": False,
            },
            "uncertainty": {
                "method": "dropout_mc",
                "mc_samples": 20,
            },
        }
        merged_config = _merge_surrogate_config(default_config, config)
        declared_inputs = [
            str(merged_config.get("input_mapping", {}).get(var_name, var_name))
            for var_name in merged_config["input_variables"]
        ]
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="ML surrogate for ozone-stress factor under uncertain detoxification capacity.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "f_O3_ML = NN(g_s_ML, T_air, VPD, O3)"
            ),
            references=[
                "Lombardozzi et al. (2015). Global Change Biology 21, 4138-4153.",
                "Sitch et al. (2007). Nature 448, 791-794.",
            ],
            parameters={
                "tutorial.case.ml_surrogate.ozone_stress.output_min": "Lower clipping bound for ozone_stress_factor_ML [-]",
                "tutorial.case.ml_surrogate.ozone_stress.output_max": "Upper clipping bound for ozone_stress_factor_ML [-]",
                "tutorial.case.ml_surrogate.ozone_stress.model_path": "Serialized PyTorch checkpoint path for ozone-stress surrogate",
            },
            input_variables=declared_inputs,
            output_variables=["ozone_stress_factor_ML"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=merged_config, metadata=metadata)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Lombardozzi et al. (2015), ozone damage scaling to plant function.
        # - Sitch et al. (2007), large-scale ozone effects on vegetation stress.
        # Mathematical expression:
        # - f_O3_ML_raw = NN(g_s_ML, T_air, VPD, O3)
        # - f_O3_ML = clip(f_O3_ML_raw, f_O3_min, f_O3_max)
        # Parameter meanings:
        # - f_O3_min/f_O3_max: admissible ozone stress factor bounds [-]
        outputs = super().forward(state, time, dt, **kwargs)
        ozone_stress_ml = outputs.get("ozone_stress_factor_ML")
        if ozone_stress_ml is None:
            return outputs
        lower = self.get_parameter_value(
            "output_min",
            namespace="tutorial.case.ml_surrogate.ozone_stress",
            required=True,
        )
        upper = self.get_parameter_value(
            "output_max",
            namespace="tutorial.case.ml_surrogate.ozone_stress",
            required=True,
        )
        lower_tensor = torch.as_tensor(
            float(lower),
            dtype=ozone_stress_ml.dtype,
            device=ozone_stress_ml.device,
        )
        upper_tensor = torch.as_tensor(
            float(upper),
            dtype=ozone_stress_ml.dtype,
            device=ozone_stress_ml.device,
        )
        outputs["ozone_stress_factor_ML"] = torch.clamp(
            ozone_stress_ml,
            min=lower_tensor,
            max=upper_tensor,
        )
        return outputs


class CaseMEGANDiurnalActivitySurrogate(_TorchGradientSafeSurrogate):
    """ML surrogate for multiplicative correction of MEGAN daytime light-temperature activity."""

    def __init__(
        self,
        name: str = "ml_diurnal_activity",
        model_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_root = Path(model_dir or Path("models"))
        default_input_variables = [
            "ppfd",
            "air_temperature",
            "leaf_area_index",
            "net_photosynthesis",
            "water_stress_factor",
            "ozone_stress_factor",
            "hour_sin",
            "hour_cos",
        ]
        default_config: Dict[str, Any] = {
            "model_path": str(model_root / "diurnal_activity_surrogate.pth"),
            "validate_model_path": False,
            "conservation_check": False,
            "uncertainty_quantification": False,
            "dependencies": ["farquhar_photosynthesis", "megan_bvoc_model"],
            "input_variables": default_input_variables,
            "input_units": {
                "hour_sin": "1",
                "hour_cos": "1",
            },
            "output_variables": ["isoprene_activity_correction_ML"],
            "output_units": {
                "isoprene_activity_correction_ML": "1",
            },
            "architecture": {
                "type": "mlp",
                "hidden_dims": [32, 16],
                "activation": "relu",
                "dropout_rate": 0.05,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 5.0e-4,
                "batch_size": 32,
                "epochs": 180,
                "loss_function": "huber",
                "early_stopping_patience": 18,
                "validation_split": 0.2,
                "shuffle_training": False,
                "daytime_par_threshold": 10.0,
            },
            "uncertainty": {
                "method": "dropout_mc",
                "mc_samples": 20,
            },
        }
        merged_config = _merge_surrogate_config(default_config, config)
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="ML surrogate for multiplicative correction of MEGAN daytime light-temperature activity.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "f_LT_ML = NN(PPFD, T_air, LAI, A_n, f_W, f_O3, sin_h, cos_h)"
            ),
            references=[
                "Guenther et al. (2012). Model of Emissions of Gases and Aerosols from Nature 2.1.",
                "Reichstein et al. (2019). Deep learning and process understanding for data-driven Earth system science.",
            ],
            parameters={
                "tutorial.case.ml_surrogate.diurnal_activity.output_min": "Lower bound for MEGAN diurnal activity correction factor [-]",
                "tutorial.case.ml_surrogate.diurnal_activity.output_max": "Upper bound for MEGAN diurnal activity correction factor [-]",
                "tutorial.case.ml_surrogate.diurnal_activity.model_path": "Serialized PyTorch checkpoint path for MEGAN diurnal activity surrogate",
            },
            input_variables=list(merged_config["input_variables"]),
            output_variables=["isoprene_activity_correction_ML"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=merged_config, metadata=metadata)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Guenther et al. (2012), daytime isoprene activity is dominated by light and temperature responses.
        # - Reichstein et al. (2019), process-level ML corrections should preserve mechanistic structure.
        # Mathematical expression:
        # - f_LT,ML = f_min + sigmoid(NN(x_day)) * (f_max - f_min)
        # Parameter meanings:
        # - f_min/f_max: lower and upper bounds of diurnal activity correction [-]
        outputs = super().forward(state, time, dt, **kwargs)
        correction = outputs.get("isoprene_activity_correction_ML")
        if correction is None:
            return outputs
        lower = self.get_parameter_value(
            "output_min",
            namespace="tutorial.case.ml_surrogate.diurnal_activity",
            required=True,
        )
        upper = self.get_parameter_value(
            "output_max",
            namespace="tutorial.case.ml_surrogate.diurnal_activity",
            required=True,
        )
        lower_tensor = torch.as_tensor(
            float(lower),
            dtype=correction.dtype,
            device=correction.device,
        )
        upper_tensor = torch.as_tensor(
            float(upper),
            dtype=correction.dtype,
            device=correction.device,
        )
        outputs["isoprene_activity_correction_ML"] = lower_tensor + torch.sigmoid(
            correction
        ) * (upper_tensor - lower_tensor)
        return outputs


class CaseIsopreneResidualSurrogate(_TorchGradientSafeSurrogate):
    """Residual MLP that scales mechanistic isoprene emission while preserving MEGAN."""

    def __init__(
        self,
        name: str = "ml_isoprene_residual",
        model_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_root = Path(model_dir or Path("models"))
        # Committed state inputs only: MEGAN's environmental stress and CO2
        # inhibition factors are diagnostics (diagnostics.bvoc.megan.*), never
        # state. The stress components water/temperature/ozone and the CO2
        # driver atmospheric_co2_ppm [umol mol-1] of the MEGAN CO2 inhibition
        # carry the same information.
        default_input_variables = [
            "isoprene_emission",
            "water_stress_factor",
            "temperature_stress_factor",
            "ozone_stress_factor",
            "stomatal_conductance",
            "net_photosynthesis",
            "atmospheric_co2_ppm",
            "ppfd",
            "air_temperature",
            "leaf_area_index",
        ]
        default_config: Dict[str, Any] = {
            "model_path": str(model_root / "isoprene_residual_surrogate.pth"),
            "validate_model_path": False,
            "conservation_check": False,
            "uncertainty_quantification": False,
            "dependencies": ["megan_bvoc_model"],
            "input_variables": default_input_variables,
            "input_units": {
                "stomatal_conductance": "mol m-2 s-1",
            },
            "output_variables": ["isoprene_correction_factor_ML"],
            "output_units": {
                "isoprene_correction_factor_ML": "1",
            },
            "architecture": {
                "type": "resnet",
                "hidden_dim": 64,
                "n_blocks": 3,
                "activation": "relu",
                "dropout_rate": 0.05,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 4.0e-4,
                "batch_size": 32,
                "epochs": 240,
                "loss_function": "huber",
                "early_stopping_patience": 24,
                "validation_split": 0.2,
                "shuffle_training": False,
            },
            "uncertainty": {
                "method": "dropout_mc",
                "mc_samples": 40,
            },
        }
        merged_config = _merge_surrogate_config(default_config, config)
        declared_inputs = [
            str(merged_config.get("input_mapping", {}).get(var_name, var_name))
            for var_name in merged_config["input_variables"]
        ]
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Residual MLP that scales mechanistic isoprene emission without replacing MEGAN.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "E_iso_hybrid = E_iso_MEGAN * f_residual_ML(x_state)"
            ),
            references=[
                "Guenther et al. (2012). Model of Emissions of Gases and Aerosols from Nature 2.1.",
                "Oumami et al. (2024). Coupled MEGAN evaluation under climate perturbations.",
            ],
            parameters={
                "tutorial.case.ml_surrogate.isoprene_residual.output_min": "Lower bound for isoprene residual correction factor [-]",
                "tutorial.case.ml_surrogate.isoprene_residual.output_max": "Upper bound for isoprene residual correction factor [-]",
                "tutorial.case.ml_surrogate.isoprene_residual.model_path": "Serialized PyTorch checkpoint path for isoprene residual surrogate",
            },
            input_variables=declared_inputs,
            output_variables=["isoprene_correction_factor_ML"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=merged_config, metadata=metadata)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Guenther et al. (2012), mechanistic MEGAN remains the primary isoprene pathway.
        # - Hybrid residual learning uses a bounded correction on top of physics.
        # Mathematical expression:
        # - f_iso = f_min + sigmoid(NN(x)) * (f_max - f_min)
        # - E_iso_hybrid = E_iso_MEGAN * f_iso
        # Parameter meanings:
        # - f_min/f_max: lower and upper bounds of multiplicative correction factor [-]
        outputs = super().forward(state, time, dt, **kwargs)
        correction_factor = outputs.get("isoprene_correction_factor_ML")
        if correction_factor is None:
            return outputs
        lower = self.get_parameter_value(
            "output_min",
            namespace="tutorial.case.ml_surrogate.isoprene_residual",
            required=True,
        )
        upper = self.get_parameter_value(
            "output_max",
            namespace="tutorial.case.ml_surrogate.isoprene_residual",
            required=True,
        )
        lower_tensor = torch.as_tensor(
            float(lower),
            dtype=correction_factor.dtype,
            device=correction_factor.device,
        )
        upper_tensor = torch.as_tensor(
            float(upper),
            dtype=correction_factor.dtype,
            device=correction_factor.device,
        )
        outputs["isoprene_correction_factor_ML"] = lower_tensor + torch.sigmoid(
            correction_factor
        ) * (upper_tensor - lower_tensor)
        return outputs

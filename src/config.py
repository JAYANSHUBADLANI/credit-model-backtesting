"""Typed access to config/config.yaml.

Same shape as the config module in the drift monitoring project this accompanies, and for the
same reason: no threshold, window length or vintage set is ever written next to the logic that
uses it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class DataConfig:
    origination_glob: str
    performance_glob: str
    delimiter: str = "|"


@dataclass
class TargetConfig:
    performance_window_months: int
    default_dlq_threshold: int
    credit_loss_zero_balance_codes: List[str]
    prepaid_zero_balance_codes: List[str]
    drop_incomplete: bool = True


@dataclass
class VintageConfig:
    fit: List[int]
    backtest: List[int]
    holdout_fraction: float
    seed: int


@dataclass
class FeatureConfig:
    numeric: List[str]
    categorical: List[str]

    @property
    def all_features(self) -> List[str]:
        return list(self.numeric) + list(self.categorical)


@dataclass
class BinningConfig:
    max_prebins: int = 20
    min_bin_fraction: float = 0.03
    min_bin_bads: int = 5
    enforce_monotonic: bool = True
    min_categorical_fraction: float = 0.01


@dataclass
class SelectionConfig:
    min_iv: float = 0.02
    max_correlation: float = 0.75


@dataclass
class ScalingConfig:
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0


@dataclass
class BandConfig:
    decline_below_percentile: float
    refer_below_percentile: float


@dataclass
class BacktestConfig:
    bootstrap_samples: int
    bootstrap_seed: int
    confidence: float
    calibration_bands: int
    min_band_count: int


@dataclass
class TriggerConfig:
    calibration_ratio_tolerance: float
    calibration_confidence: float
    persistence_vintages: int
    cooldown_vintages: int


@dataclass
class ChallengerConfig:
    training_lag_years: int
    training_window_years: int
    min_training_rows: int


@dataclass
class StabilityConfig:
    reference_bins: int
    psi_warn: float
    psi_alert: float
    csi_warn: float
    csi_alert: float


@dataclass
class ArtifactConfig:
    model_path: str
    reference_path: str
    reports_dir: str


@dataclass
class Config:
    data: DataConfig
    target: TargetConfig
    vintages: VintageConfig
    features: FeatureConfig
    binning: BinningConfig
    selection: SelectionConfig
    scaling: ScalingConfig
    bands: BandConfig
    backtest: BacktestConfig
    triggers: TriggerConfig
    challenger: ChallengerConfig
    stability: StabilityConfig
    artifacts: ArtifactConfig
    root: Path = field(default=PROJECT_ROOT)

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.root / candidate


def _derive_root(config_path: Path) -> Path:
    """Work out the project root that relative config paths resolve against.

    Guessing it is worse than refusing to. A wrong root that silently works, creating the tree
    it points at on first write, is far harder to find than one that fails on load.
    """
    if config_path.parent.name == "config":
        return config_path.parents[1]
    raise ValueError(
        f"cannot derive the project root from {config_path}. Put the file at "
        "<root>/config/<name>.yaml, or pass root= to load_config, or set PROJECT_ROOT."
    )


def _as_dict(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    section = raw.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"config section '{key}' is missing or malformed")
    return section


def load_config(path: Path | str | None = None, root: Path | str | None = None) -> Config:
    resolved = Path(path or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIG_PATH).resolve()
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    override = root or os.environ.get("PROJECT_ROOT")
    resolved_root = Path(override).resolve() if override else _derive_root(resolved)

    config = Config(
        data=DataConfig(**_as_dict(raw, "data")),
        target=TargetConfig(**_as_dict(raw, "target")),
        vintages=VintageConfig(**_as_dict(raw, "vintages")),
        features=FeatureConfig(**_as_dict(raw, "features")),
        binning=BinningConfig(**_as_dict(raw, "binning")),
        selection=SelectionConfig(**_as_dict(raw, "selection")),
        scaling=ScalingConfig(**_as_dict(raw, "scaling")),
        bands=BandConfig(**_as_dict(raw, "bands")),
        backtest=BacktestConfig(**_as_dict(raw, "backtest")),
        triggers=TriggerConfig(**_as_dict(raw, "triggers")),
        challenger=ChallengerConfig(**_as_dict(raw, "challenger")),
        stability=StabilityConfig(**_as_dict(raw, "stability")),
        artifacts=ArtifactConfig(**_as_dict(raw, "artifacts")),
        root=resolved_root,
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    """Refuse a config that would produce a plausible but meaningless backtest."""
    missing = [v for v in config.vintages.fit if v not in config.vintages.backtest]
    if missing:
        raise ValueError(
            f"fit vintages {missing} are not in the backtest set. The fit period has to sit on "
            "the same axis as everything else or there is no day one number to compare against."
        )
    if config.target.performance_window_months <= 0:
        raise ValueError("performance_window_months must be positive")
    if not 0.0 < config.vintages.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    overlap = set(config.target.credit_loss_zero_balance_codes) & set(
        config.target.prepaid_zero_balance_codes
    )
    if overlap:
        raise ValueError(
            f"zero balance codes {sorted(overlap)} are listed as both a credit loss and a "
            "voluntary prepayment. One loan cannot be both and the target would depend on "
            "evaluation order."
        )

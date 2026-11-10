"""Typed public contracts; no dataset paths, coordinates or future labels."""
from typing import Literal

from pydantic import BaseModel,ConfigDict,Field,model_validator


class ReplayRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    sensor_id: str = Field(min_length=1,max_length=64,pattern=r'^synthetic_sensor_[0-9]+$')
    index: int | None = Field(default=None,strict=True,ge=0)
    timestamp: str | None = Field(default=None,max_length=32,pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$')
    scenario: Literal['clean','missing','noisy','stuck'] = 'clean'
    fault_rate: Literal[0.1,0.2,0.4] = 0.4
    seed: Literal[1729,2718,31415] = 1729
    policy: Literal['controller','xgboost'] = 'controller'

    @model_validator(mode='after')
    def exactly_one_origin(self):
        if (self.index is None)==(self.timestamp is None):
            raise ValueError('Specify exactly one replay index or timestamp')
        return self


class PublicResponse(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)


class Reading(PublicResponse):
    index: int
    relative_minutes: int
    timestamp: str
    value: float | None
    usable: bool
    filled_value: float


class Interval(PublicResponse):
    status: Literal['calibration_derived','unavailable']
    available: bool
    lower: float | None = None
    upper: float | None = None
    nominal_coverage: float | None = None
    reason: str

    @model_validator(mode='after')
    def consistent_availability(self):
        if self.available:
            if self.status!='calibration_derived' or self.lower is None or self.upper is None or self.nominal_coverage is None or self.lower>self.upper:
                raise ValueError('Available intervals require ordered calibration-derived bounds')
        elif self.status!='unavailable' or any(v is not None for v in (self.lower,self.upper,self.nominal_coverage)):
            raise ValueError('Unavailable intervals must not contain bounds or coverage')
        return self


class Forecast(PublicResponse):
    horizon_steps: Literal[3,6,12]
    horizon_minutes: Literal[15,30,60]
    timestamp: str
    prediction: float | None
    selected_model: str
    interval: Interval


class Quality(PublicResponse):
    unavailable_fraction: float
    repeated_pair_fraction: float
    abrupt_pair_fraction: float
    usable_adjacent_pair_count: int


class Decision(PublicResponse):
    choice: Literal['primary','fallback','abstain']
    reason: Literal['primary_unavailable','both_unavailable','fallback_unavailable','always_primary',
                    'input_unavailable','model_disagreement','within_thresholds','fixed_xgboost_policy']
    quality: Quality
    disagreement_sigma: float | None
    interval_validity: str


class FaultSummary(PublicResponse):
    kind: Literal['missing','noisy','stuck']
    mode: Literal['point','contiguous']
    rate: float
    seed: int
    sampling_policy: Literal['synthetic_ceil']
    achieved_selection_rate: float
    selected_cell_count: int
    changed_cell_count: int
    scope: str


class Playback(PublicResponse):
    previous_index: int | None
    next_index: int | None
    first_index: int
    last_index: int


class ReplayResponse(PublicResponse):
    contract_version: Literal['1.1']
    data_origin: str
    mode: Literal['synthetic_historical_replay']
    units: str
    sensor_id: str
    sensor_label: str
    index: int
    timestamp: str
    scenario: str
    policy: str
    joint_decision_scope: str
    decision: Decision
    history: list[Reading] = Field(min_length=12,max_length=12)
    forecasts: list[Forecast] = Field(min_length=3,max_length=3)
    fault_summary: FaultSummary | None
    playback: Playback


class HealthResponse(PublicResponse):
    status: str
    synthetic_only: bool
    ready: bool


class SensorDetail(PublicResponse):
    sensor_id: str
    label: str
    synthetic: Literal[True]


class HorizonDetail(PublicResponse):
    steps: Literal[3,6,12]
    minutes: Literal[15,30,60]
    label: str


class ReplayAvailability(PublicResponse):
    available: Literal[True]
    partition: str
    first_index: int
    last_index: int
    first_timestamp: str
    last_timestamp: str
    origin_count: int
    index_stride: Literal[1]
    history_steps: Literal[12]
    interval_minutes: Literal[5]
    timestamp_convention: str
    final_test_available: Literal[False]


class Geography(PublicResponse):
    available: Literal[False]
    reason: Literal['no_geographic_coordinates_provided']


class MetadataResponse(PublicResponse):
    contract_version: Literal['1.1']
    data_origin: str
    mode: str
    units: str
    sensors: list[str]
    sensor_details: list[SensorDetail]
    models: list[str]
    horizons_steps: list[int]
    horizons_minutes: list[int]
    supported_horizons: list[HorizonDetail]
    replay: ReplayAvailability
    geography: Geography
    scenarios: list[str]
    interval_policy: str
    cpu_threads_max: int
    policies: list[Literal['controller','xgboost']]
    fault_rates: list[float]
    fault_seeds: list[int]

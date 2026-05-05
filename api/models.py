from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Union


class PreliminaryTestResponse(BaseModel):
    status: int
    message: str
    warnings: Optional[List[str]] = None


class DataFrameInput(BaseModel):
    data: List[dict]


class UniqueIDResponse(BaseModel):
    UniqueID: List[str]
    Numeric_DataTypes: int


class UniqueIDCheckInput(BaseModel):
    data: List[dict]
    columns: List[str]


class UniqueIDCheckResponse(BaseModel):
    result: Tuple[str, bool]


class FileUpload(BaseModel):
    file: bytes


class DropExportDuplicatesInput(BaseModel):
    uidCol: Union[str, List[str]]
    keptRow: str = "first"
    export: bool = True
    chunksize: Optional[int] = None


class DropExportDuplicatesResponse(BaseModel):
    unique_count: int
    duplicate_count: int
    percent_duplicates: float


class MissingEntriesInput(BaseModel):
    column_to_analyze: str
    group_by: Optional[str] = None
    filter_by: Optional[Dict[str, str]] = None


class ErrorHandlingInput(BaseModel):
    params: dict


class L1SampleSizeInput(BaseModel):
    min_n_samples: int
    max_n_samples: int
    n_subs_per_block: int
    n_blocks_per_district: int
    n_district: int
    level_test: str
    percent_punish: float
    percent_guarantee: float
    confidence: float
    n_simulations: int
    min_disc: float
    max_disc: float
    mean_disc: float
    std_disc: float
    distribution: str


class L2SampleSizeInput(BaseModel):
    total_samples: int
    average_truth_score: float
    sd_across_blocks: float
    sd_within_block: float
    level_test: str
    n_subs_per_block: int
    n_blocks_per_district: int
    n_district: int
    n_simulations: int
    min_sub_per_block: int


class ThirdPartySamplingInput(BaseModel):
    total_samples: int
    average_truth_score: float
    sd_across_blocks: float
    sd_within_block: float
    level_test: str
    n_subs_per_block: int
    n_blocks_per_district: int
    n_district: int
    n_simulations: int
    min_sub_per_block: int
    percent_blocks_plot: float
    errorbar_type: str
    n_blocks_reward: int

# 定义了Pydantic模式，用于数据验证和序列化

from pydantic import BaseModel, BaseSettings
from typing import Optional, List
from datetime import date, datetime

class UserLog(BaseModel):
    user_id: int
    username: str
    action: str
    timestamp: datetime
    details: Optional[str] = None  # 新增字段记录详细信息

    class Config:
        orm_mode = True

class Settings(BaseSettings):
    authjwt_secret_key: str = "J6g3Fx1VqIVO48yIuOIEgVtSDZmQKY6nuN1"  
    # JWT密钥
    authjwt_algorithm: str = "HS256"

    authjwt_access_token_expires: int = 60 * 60 * 24 # 访问令牌有效期 1 days
    # authjwt_refresh_token_expires: int = 60 * 24 * 30  # 刷新令牌有效期 30 days

    class Config:
        env_file = ".env"

class RegisterModel(BaseModel):
    username: str
    password: str

class LoginModel(BaseModel):
    username: str
    password: str

class Report(BaseModel):
    report_date: date
    content: str

    class Config:
        orm_mode = True

# 定义一个基础模型 HumanSingleCellTrackingTableBase，用于存储公共字段
class HumanSingleCellTrackingTableBase(BaseModel):
    # id: int
    PTRSB: Optional[str] = None
    injection_id: Optional[str] = None
    cell_id: str
    # tissue_id: Optional[str] = None
    # slice_id: Optional[str] = None
    patient_number: Optional[str] = None
    tissue_block_number: Optional[str] = None
    small_number: Optional[str] = None
    slice_number: Optional[str] = None
    confirmed: Optional[str] = None
    slicing_method: Optional[str] = None
    slicing_and_fixation_order: Optional[str] = None
    pre_fixation_days: Optional[str] = None
    post_fixation_days: Optional[str] = None
    sample_preparation_staff: Optional[str] = None
    sample_preparation_date: Optional[str] = None
    fresh_perfusion: Optional[str] = None
    post_perfusion_4percent_pfa: Optional[str] = None
    post_perfusion_10percent_formalin: Optional[str] = None
    brain_region: Optional[str] = None
    slice_thickness: Optional[str] = None
    confirmed_0: Optional[str] = None
    tissue_dissection_time: Optional[str] = None
    perfusion_start_time: Optional[str] = None
    perfusion_end_time: Optional[str] = None
    after_surgery_hours: Optional[str] = None
    cell_depth: Optional[str] = None
    perfusion_current: Optional[str] = None
    perfusion_time_on: Optional[str] = None
    perfusion_time_off: Optional[str] = None
    dye_name: Optional[str] = None  ## 新增
    dye_concentration: Optional[str] = None
    experiment_temperature: Optional[str] = None
    experiment_humidity: Optional[str] = None
    inject_method: Optional[str] = None  # 新增
    perfusion_date: Optional[str] = None
    perfusion_staff: Optional[str] = None
    ihc_category: Optional[str] = None
    immunohistochemistry: Optional[str] = None
    first_antibody_concentration: Optional[str] = None
    secondary_antibody_band: Optional[str] = None
    image_device: Optional[str] = None
    dapi_concentration: Optional[str] = None
    laser_wavelength: Optional[str] = None
    laser_power: Optional[str] = None
    laser_power_ratio: Optional[str] = None
    gain: Optional[str] = None   ### 新增
    scanner: Optional[str] = None    ### 新增
    averaging: Optional[str] = None    ### 新增
    pmt_voltage: Optional[str] = None
    z_size: Optional[str] = None
    tiling: Optional[str] = None
    overlap: Optional[str] = None
    xy_resolution: Optional[str] = None
    z_resolution: Optional[str] = None
    document_name: Optional[str] = None
    image_cell_id: Optional[str] = None  # 新增
    shooting_date: Optional[str] = None
    shooting_staff: Optional[str] = None
    image_size: Optional[str] = None  ## 新增
    confirmed_1: Optional[str] = None
    reconstruction_staff:Optional[str] = None
    status: Optional[str] = None
    inspection_staff: Optional[str] = None
    status_0: Optional[str] = None
    sealed_slide: Optional[str] = None
    status_1: Optional[str] = None
    dye_solvent: Optional[str] = None
    remarks: Optional[str] = None
    image_file: Optional[str] = None
    v3dpbd_file: Optional[str] = None  # 新增
    soma_x: Optional[str] = None  # 新增
    soma_y: Optional[str] = None  # 新增
    soma_z: Optional[str] = None  # 新增
    swc_auto14: Optional[str] = None

    class Config:
        orm_mode = True  # 设置 from_attributes 为 True

# 定义一个创建模型 HumanSingleCellTrackingTableCreate，继承 HumanSingleCellTrackingTableBase
# 这个模型用于创建新的单细胞数据
class HumanSingleCellTrackingTableCreate(HumanSingleCellTrackingTableBase):
    pass

# 定义一个响应模型 HumanSingleCellTrackingTable，继承 HumanSingleCellTrackingTableBase
# 这个模型用于返回数据库中的单细胞数据
class HumanSingleCellTrackingTable(HumanSingleCellTrackingTableBase):
    id: int

    class Config:
        from_attributes = True
        # orm_mode = True    # 启用 ORM 模式，允许 Pydantic 模型与 SQLAlchemy 模型进行交互

class SampleInfoBase(BaseModel):
    # idx: int
    total_id: str
    patient_id: Optional[str] = None
    sample_id: Optional[str] = None
    id: Optional[str] = None
    patient_number: Optional[str] = None
    tissue_id: Optional[str] = None
    sample_slice_info: Optional[str] = None
    perfusion_info: Optional[str] = None
    perfusion_evaluation: Optional[str] = None
    patient_age: Optional[str] = None
    gender: Optional[str] = None
    surgery_date: Optional[str] = None
    sample_size: Optional[str] = None
    tumor_location: Optional[str] = None
    intracranial_location: Optional[str] = None
    english_abbr_nj: Optional[str] = None
    english_full_name: Optional[str] = None
    left_right_brain: Optional[str] = None
    sample_processing_method: Optional[str] = None
    removal_time: Optional[str] = None
    immersion_time: Optional[str] = None
    storage_location: Optional[str] = None
    sampling_method_personnel: Optional[str] = None
    send_sample_date: Optional[str] = None
    treatment_naive: Optional[str] = None
    treatment_received: Optional[str] = None
    pathological_diagnosis: Optional[str] = None

    sample_snapshot: Optional[str] = None
    sample_image: Optional[str] = None
    sample_annotation: Optional[str] = None

    class Config:
        orm_mode = True

# 创建模型
class SampleInfoCreate(SampleInfoBase):
    pass

# 响应模型
class SampleInfo(SampleInfoBase):
    idx: int

    class Config:
        from_attributes = True


class ImagingRecordSchema(BaseModel):
    id: Optional[int]
    imaging_id: str
    sample_preparation_id: int
    producer: str
    status: str
    Dyes: Optional[int] = 1  # 默认值为 1
    Z_Size: Optional[float] = None
    Y_Size: Optional[float] = None
    X_Size: Optional[float] = None
    File_Size_GB: Optional[float] = None
    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()  # 自动转换 datetime
        }

class SamplePreparationSchema(BaseModel):
    id: Optional[int]
    sampleId: str
    tissueId: str
    rollId: str
    sliceId: str
    blockId: str
    dyes: int
    needles: int
    injected_num:int
    dye_name: Optional[str]
    perfusion_user: Optional[str]
    perfusion_date: Optional[str]
    status: str
    comment: Optional[str] = None
    imaging_records: Optional[List[ImagingRecordSchema]]
    created_at: Optional[datetime]
    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()  # 自动将 datetime 转换为字符串
        }
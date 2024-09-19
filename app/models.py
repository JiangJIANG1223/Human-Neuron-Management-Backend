# 定义了数据库模型HumanSingleCellTrackingTable，该模型与MySQL中的Human_SingleCell_TrackingTable表对应

from sqlalchemy import Column, String, Integer, Text, Date, TIMESTAMP, func, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column("username", String, unique=True, index=True)
    hashed_password = Column("hashed_password", String)

    logs = relationship("UserLog", back_populates="user")

class UserLog(Base):
    __tablename__ = 'user_logs_20240725'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String)
    action = Column("action", String, index=True)
    timestamp = Column("timestamp", DateTime, default=datetime.utcnow)
    details = Column("details", Text)  # 新增字段记录详细信息

    user = relationship("User", back_populates="logs")
    
class HumanSingleCellTrackingTable(Base):
    __tablename__ = "human_singlecell_trackingtable_20240712"
    id = Column(Integer, primary_key=True, autoincrement=True)
    cell_id = Column("Cell ID", String(255))
    # tissue_id = Column("Tissue ID", String(255))  # 删除
    # slice_id = Column("Silce ID", String(255))  # 删除
    patient_number = Column("病人编号", String(255))  # Smaple ID
    tissue_block_number = Column("组织块编号", String(255))  # Tissue ID
    small_number = Column("小编号", String(255))
    slice_number = Column("切片编号", String(255))  # Slice ID
    confirmed = Column("confirmed", String(255))
    slicing_method = Column("切片方式(0:振动切片;1:冰冻切片)", String(255))
    slicing_and_fixation_order = Column("切片与固定顺序(0:先固定;1:先切片)", String(255))
    pre_fixation_days = Column("切片前固定天数", String(255))
    post_fixation_days = Column("切片后固定天数", String(255))
    sample_preparation_staff = Column("样本制备人员", String(255))
    sample_preparation_date = Column("样本制备日期", String(255))
    fresh_perfusion = Column("新鲜灌注(0:否;1:是)", String(255))
    post_perfusion_4percent_pfa = Column("4%多聚甲醛固定后灌注(0:否;1:是)", String(255))
    post_perfusion_10percent_formalin = Column("10%甲醛固定后灌注(0:否;1:是)", String(255))
    brain_region = Column("脑区", String(255))
    slice_thickness = Column("切片厚度(微米)", String(255))
    confirmed_0 = Column("confirmed_[0]", String(255))
    tissue_dissection_time = Column("组织离体时间", String(255))
    perfusion_start_time = Column("灌注开始时间(e.g.14:30)", String(255))
    perfusion_end_time = Column("灌注结束时间(e.g.18:30)", String(255))
    after_surgery_hours = Column("after surgery(h)", String(255))
    cell_depth = Column("细胞深度(微米)", String(255))
    perfusion_current = Column("灌注电流(nA)", String(255))
    perfusion_time_on = Column("灌注时间(通电,mins)", String(255))
    perfusion_time_off = Column("灌注时间(未通电,mins)", String(255))
    dye_name = Column("染料名称", String(255))    ## 新增列
    dye_concentration = Column("染料浓度(%)", String(255))
    experiment_temperature = Column("实验温度(℃)", String(255))
    experiment_humidity = Column("实验湿度(%RH)", String(255))
    inject_method = Column("Manual/Auto Inject(0:Manual;1:Auto)", String(255))  # 新增列
    perfusion_date = Column("灌注日期", String(255))
    perfusion_staff = Column("灌注人员", String(255))
    ihc_category = Column("类别", String(255))  ## 新增
    immunohistochemistry = Column("免疫染色(0:否;1:是)", String(255))
    first_antibody_concentration = Column("一抗浓度", String(255))
    secondary_antibody_band = Column("二抗波段", String(255))
    dapi_concentration = Column("DAPI浓度", String(255))
    laser_wavelength = Column("激光波长(nm)", String(255))
    laser_power = Column("激光功率(mW)", String(255))
    laser_power_ratio = Column("激光功率比例(%)", String(255))
    pmt_voltage = Column("PMT(V)", String(255))
    z_size = Column("Z_size(张)", String(255))
    tiling = Column("平铺(0:否;1:是)", String(255))
    overlap = Column("overlap", String(255))
    xy_resolution = Column("xy拍摄分辨率(*10e-3μm/px)", String(255))
    z_resolution = Column("z拍摄分辨率(*10e-3μm/px)", String(255))
    document_name = Column("Document Name", String(255))
    image_cell_id = Column("Image Cell ID", String(255))  # 新增列
    shooting_date = Column("拍摄日期", String(255))
    shooting_staff = Column("拍摄人员", String(255))
    image_size = Column("Image Size", String(255))  ## 新增列
    confirmed_1 = Column("confirmed_[1]", String(255))
    reconstruction_staff = Column("重建人员", String(255))
    status = Column("status", String(255))
    inspection_staff = Column("检查人员", String(255))
    status_0 = Column("status_[0]", String(255))
    sealed_slide = Column("是否封片", String(255))
    status_1 = Column("status_[1]", String(255))
    dye_solvent = Column("染料溶剂", Text)
    remarks = Column("备注", Text)
    image_file = Column("image_file", Text)
    v3dpbd_file = Column("v3dpbd_file", Text)
    soma_x = Column("soma_x", String(255))
    soma_y = Column("soma_y", String(255))
    soma_z = Column("soma_z", String(255))
    swc_auto14 = Column("swc_auto14", String(255))


class Sample_Information(Base):
    __tablename__ = "sample_information_20240815"
    idx = Column(Integer, primary_key=True, autoincrement=True)

    total_id = Column("总表编号", String(255))
    patient_id = Column("患者编号", String(255))
    sample_id = Column("样本编号", String(255))
    id = Column("ID", String(255))
    patient_number = Column("病人编号", String(255))
    tissue_id = Column("组织编号", String(255))
    sample_slice_info = Column("样本切片情况", String(255))
    perfusion_info = Column("灌注情况", String(255))
    perfusion_evaluation = Column("灌注效果评价", String(255))
    patient_age = Column("患者年龄(取材时)", String(255))
    gender = Column("性别", String(255))
    surgery_date = Column("手术日期", String(255))
    sample_size = Column("样本大小", String(255))
    tumor_location = Column("肿瘤部位", String(255))
    intracranial_location = Column("颅内取材部位", String(255))
    english_abbr_nj = Column("英文简称(南京编)", String(255))
    english_full_name = Column("英文全称", String(255))
    left_right_brain = Column("左/右脑", String(255))
    sample_processing_method = Column("样本处理方式", String(255))
    removal_time = Column("取下时间", String(255))
    immersion_time = Column("样本浸液时间", String(255))
    storage_location = Column("样本存放地点", String(255))
    sampling_method_personnel = Column("取样方式/人员", String(255))
    send_sample_date = Column("寄出/取样日期", String(255))
    treatment_naive = Column("Treatment Naive", String(255))
    treatment_received = Column("接受过何种治疗", Text)
    pathological_diagnosis = Column("病理诊断", String(255))

class DailyReport(Base):
    __tablename__ = 'daily_reports_20240628'
    id = Column(Integer, primary_key=True, index=True)
    report_date = Column(Date, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    

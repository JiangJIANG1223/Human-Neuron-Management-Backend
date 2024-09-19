# 定义了基本的数据库操作函数，例如get_single_cell_data用于查询属性数据, get_total_count获取总记录数
from fastapi import HTTPException
from sqlalchemy.orm import Session
from . import models, schemas
from sqlalchemy import func, cast, Integer
from datetime import datetime
import json
import pytz

def get_single_cell_data(db: Session, skip: int = 0, limit: int = 10, query_params=None):
    query = db.query(models.HumanSingleCellTrackingTable)
    
    if query_params:
        if query_params.get('cell_id_start'):
            query = query.filter((cast(models.HumanSingleCellTrackingTable.cell_id, Integer)) >= query_params['cell_id_start'])
        if query_params.get('cell_id_end'):
            query = query.filter((cast(models.HumanSingleCellTrackingTable.cell_id, Integer)) <= query_params['cell_id_end'])
        if query_params.get('patient_number'):
            query = query.filter(models.HumanSingleCellTrackingTable.patient_number.in_(query_params['patient_number']))    
        if query_params.get('tissue_block_number'):
            query = query.filter(models.HumanSingleCellTrackingTable.tissue_block_number.in_(query_params['tissue_block_number']))
        if query_params.get('slice_number'):
            query = query.filter(models.HumanSingleCellTrackingTable.slice_number.in_(query_params['slice_number']))
        if query_params.get('slicing_method'):
            query = query.filter(models.HumanSingleCellTrackingTable.slicing_method == query_params['slicing_method'])
        if query_params.get('fresh_perfusion'):
            query = query.filter(models.HumanSingleCellTrackingTable.fresh_perfusion == query_params['fresh_perfusion'])
        if query_params.get('brain_region'):
            query = query.filter(models.HumanSingleCellTrackingTable.brain_region.in_(query_params['brain_region']))
        if query_params.get('immunohistochemistry'):
            query = query.filter(models.HumanSingleCellTrackingTable.immunohistochemistry == query_params['immunohistochemistry'])
        if query_params.get('inject_method'):
            query = query.filter(models.HumanSingleCellTrackingTable.inject_method == query_params['inject_method'])

    return query.offset(skip).limit(limit).all()

def get_total_count(db: Session, query_params=None):
    query = db.query(func.count(models.HumanSingleCellTrackingTable.cell_id))
    
    if query_params:
        if query_params.get('cell_id_start'):
            query = query.filter((cast(models.HumanSingleCellTrackingTable.cell_id, Integer)) >= query_params['cell_id_start'])
        if query_params.get('cell_id_end'):
            query = query.filter((cast(models.HumanSingleCellTrackingTable.cell_id, Integer)) <= query_params['cell_id_end'])
        if query_params.get('patient_number'):
            query = query.filter(models.HumanSingleCellTrackingTable.patient_number.in_(query_params['patient_number']))
        if query_params.get('tissue_block_number'):
            query = query.filter(models.HumanSingleCellTrackingTable.tissue_block_number.in_(query_params['tissue_block_number']))
        if query_params.get('slice_number'):
            query = query.filter(models.HumanSingleCellTrackingTable.slice_number.in_(query_params['slice_number']))
        if query_params.get('slicing_method'):
            query = query.filter(models.HumanSingleCellTrackingTable.slicing_method == query_params['slicing_method'])
        if query_params.get('fresh_perfusion'):
            query = query.filter(models.HumanSingleCellTrackingTable.fresh_perfusion == query_params['fresh_perfusion'])
        if query_params.get('brain_region'):
            query = query.filter(models.HumanSingleCellTrackingTable.brain_region.in_(query_params['brain_region']))
        if query_params.get('immunohistochemistry'):
            query = query.filter(models.HumanSingleCellTrackingTable.immunohistochemistry == query_params['immunohistochemistry'])
        if query_params.get('inject_method'):
            query = query.filter(models.HumanSingleCellTrackingTable.inject_method == query_params['inject_method'])

    return query.scalar()

def get_default_values(db: Session):
    last_record = db.query(models.HumanSingleCellTrackingTable).order_by(models.HumanSingleCellTrackingTable.cell_id.desc()).first()
    
    if last_record:
        last_cell_id = last_record.cell_id
        next_cell_id = str(int(last_cell_id) + 1).zfill(5)  # 将cell_id填充为5位
    else:
        next_cell_id = '00001'  # 如果没有记录，初始化为'00001'
    
    last_record_dict = last_record.__dict__ if last_record else {}
    if last_record_dict:
        last_record_dict.pop('_sa_instance_state', None)
    
    return {
        "cell_id": next_cell_id,
        "last_record": last_record_dict
    }

from sqlalchemy import cast, Integer

def get_default_info(db: Session):
    last_record = db.query(models.Sample_Information).filter(models.Sample_Information.total_id != '--').order_by(cast(models.Sample_Information.total_id, Integer).desc()).first()
    
    last_record_dict = last_record.__dict__ if last_record else {}
    if last_record_dict:
        last_record_dict.pop('_sa_instance_state', None)
    
    return {
        "last_record": last_record_dict
    }

def get_total_samples(db: Session):
    return db.query(func.max(cast(models.Sample_Information.total_id, Integer))).scalar()

def get_valid_samples(db: Session):
    valid_samples = db.query(models.Sample_Information.patient_number).filter(
        models.Sample_Information.patient_number != '',
        models.Sample_Information.patient_number != '-',
        models.Sample_Information.patient_number != '--',
    ).distinct().count()
    return valid_samples

def get_total_regions(db: Session):
    return db.query(models.HumanSingleCellTrackingTable.brain_region).distinct().count()
    # return db.query(models.Sample_Information.english_abbr_nj).filter(
    #     models.Sample_Information.english_abbr_nj != '--').distinct().count()

def get_production_trend(db: Session, start_date: str, end_date: str):
    trend_data = db.query(
        models.HumanSingleCellTrackingTable.perfusion_date, 
        func.count(models.HumanSingleCellTrackingTable.cell_id)
    ).filter(
        models.HumanSingleCellTrackingTable.perfusion_date.between(start_date, end_date)
    ).group_by(
        models.HumanSingleCellTrackingTable.perfusion_date
    ).all()

    min_date = db.query(func.min(models.HumanSingleCellTrackingTable.perfusion_date)).scalar()
    max_date = db.query(func.max(models.HumanSingleCellTrackingTable.perfusion_date)).scalar()

    return trend_data, min_date, max_date

def create_single_cell_data(db: Session, data: schemas.HumanSingleCellTrackingTableCreate):
    db_data = models.HumanSingleCellTrackingTable(**data.dict())
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

def delete_single_cell_data(db: Session, cell_id: str):
    db_data = db.query(models.HumanSingleCellTrackingTable).filter(models.HumanSingleCellTrackingTable.cell_id == cell_id).first()
    if db_data:
        db.delete(db_data)
        db.commit()
        return db_data
    else:
        raise HTTPException(status_code=404, detail="Data not found")
    
def update_single_cell_data(db: Session, db_obj: models.HumanSingleCellTrackingTable, update_data: schemas.HumanSingleCellTrackingTableCreate):
    for key, value in update_data.dict().items():
        setattr(db_obj, key, value)
    db.commit()
    db.refresh(db_obj)
    return db_obj
    
def get_single_cell_data_by_id(db: Session, cell_id: str):
    return db.query(models.HumanSingleCellTrackingTable).filter(models.HumanSingleCellTrackingTable.cell_id == cell_id).first()

def create_user_log(db: Session, user_id: int, action: str, details: dict = None):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(beijing_tz)

    #  使用 ensure_ascii=False 来确保中文字符不会被转义，随后使用 encode('utf-8') 将字符串编码为 UTF-8 字节字符串。
    log_details = json.dumps(details, ensure_ascii=False).encode('utf-8') if details else None 
    log = models.UserLog(user_id=user_id, username=user.username, action=action, timestamp=current_time, details=log_details)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
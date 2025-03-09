import pandas as pd
import numpy as np
import sqlalchemy as db
from sqlalchemy import create_engine, text
from datetime import datetime
import re

# MySQL 连接设置
# DATABASE_URI = 'mysql+pymysql://root:braintell%40seu@localhost/human_neuron'
DATABASE_URI = "mysql+pymysql://root:hneuronbyseu123@localhost/human_neuron"
# DATABASE_URI = 'mysql+pymysql://root:wlj990521@10.194.35.182/hndb'
engine = db.create_engine(DATABASE_URI)
connection = engine.connect()

# 1. 从 imaging_information_20241023 和 injection_table_20241028 中提取数据
def extract_imaging_and_injection_data(ptrsb):
    # 首先，获取满足条件的 PTRS(B)
    ptrs_b_query = """
    SELECT `PTRS(B)` 
    FROM injection_table_20241028
    GROUP BY `PTRS(B)` 
    HAVING COUNT(DISTINCT dye_name) > 1
    """
    ptrs_b_result = connection.execute(text(ptrs_b_query))
    ptrs_b_list = [row[0] for row in ptrs_b_result.fetchall()]
    
    # 确保 ptrs_b_list 不为空
    if not ptrs_b_list:
        print("没有满足条件的 PTRS(B)。")
        return pd.DataFrame()  # 返回空的 DataFrame

    # 将 PTRS(B) 列表转换为用于 SQL 查询的字符串
    # 为防止 SQL 注入，使用参数化查询
    placeholders = ','.join([':ptrs_b{}'.format(i) for i in range(len(ptrs_b_list))])
    params = {'ptrs_b{}'.format(i): ptrs_b_list[i] for i in range(len(ptrs_b_list))}

    params['idx_threshold'] = 12961  # 添加 idx 的参数
    
    # 修改 imaging_query，增加条件
    imaging_query = f"""
    SELECT 
        idx AS imaging_idx,
        `PTRS(B)`,
        ID AS injection_ID,
        metadata_file,
        apo_file,  
        imaging_device,
        laser_wavelength, 
        laser_power, 
        laser_power_ratio, 
        gain, 
        scanner, 
        averaging, 
        pmt_voltage, 
        z_size, 
        tiling, 
        overlap, 
        xy_resolution, 
        z_resolution,
        document_name, 
        image_size, 
        shooting_date, 
        shooting_staff, 
        soma_x, 
        soma_y,
        soma_z
    FROM imaging_information_20241023
    WHERE `PTRS(B)` = :ptrsb
    """
    injection_query = f"""
    SELECT 
        `PTRS(B)`,
        file_name AS injection_file, 
        sample_preparation_date,
        sample_preparation_staff, 
        fresh_perfusion, 
        slice_thickness, 
        experiment_temperature,  
        perfusion_user AS perfusion_staff        
    FROM injection_table_20241028
    WHERE `PTRS(B)` IN ({placeholders})
    """
    # 执行查询，传入参数
    imaging_result = connection.execute(text(imaging_query), {"ptrsb": ptrsb})
    imaging_df = pd.DataFrame(imaging_result.fetchall(), columns=imaging_result.keys())
    
    injection_result = connection.execute(text(injection_query), params)
    injection_df = pd.DataFrame(injection_result.fetchall(), columns=injection_result.keys())
    
    # 确保 'PTRS(B)' 列的数据类型一致
    imaging_df['PTRS(B)'] = imaging_df['PTRS(B)'].astype(str)
    injection_df['PTRS(B)'] = injection_df['PTRS(B)'].astype(str)
    
    # 检查 injection_df 中相同 PTRS(B) 的注射信息是否一致
    # 去除 'PTRS(B)' 列和 'injection_file' 列，用于比较
    injection_info_columns = injection_df.columns.difference(['PTRS(B)', 'injection_file'])
    inconsistent_injections = injection_df.groupby('PTRS(B)')[injection_info_columns.tolist()].nunique()
    inconsistent_ptrs = inconsistent_injections[(inconsistent_injections > 1).any(axis=1)].index.tolist()
    
    if inconsistent_ptrs:
        print("警告：以下 PTRS(B) 存在不一致的灌注信息：")
        # 对每个存在不一致的 PTRS(B)，找出具体不一致的列
        for ptrs_b in inconsistent_ptrs:
            cols = inconsistent_injections.loc[ptrs_b]
            inconsistent_cols = cols[cols > 1].index.tolist()
            print(f"PTRS(B): {ptrs_b}, 不一致的列: {inconsistent_cols}")
        # 根据需求处理不一致的数据，例如抛出异常或手动处理
        # 在此示例中，我们选择抛出异常
        raise ValueError(f"灌注信息不一致，无法合并。请检查 PTRS(B)：{inconsistent_ptrs}")

    else:
        # 灌注信息一致，可以安全地去重
        injection_df_unique = injection_df.drop_duplicates(subset='PTRS(B)')
    
    # 以 imaging_df 为基准，基于 PTRS(B) 列合并
    merged_df = pd.merge(imaging_df, injection_df_unique, on='PTRS(B)', how='left')
    
    # 检查是否有未匹配的记录
    unmatched_ptrs_b = imaging_df[~imaging_df['PTRS(B)'].isin(injection_df_unique['PTRS(B)'])]['PTRS(B)'].unique()
    if len(unmatched_ptrs_b) > 0:
        print("警告：以下 PTRS(B) 在 injection_df 中未找到匹配的记录：")
        print(unmatched_ptrs_b)
    
    return merged_df

# 2. 从 sample_information_20240815 中提取 brain_region 和 tissue_dissection_time
def extract_sample_information(merged_df):
    sample_query = """
    SELECT 
        `病人编号`, `组织编号`, `英文简称(南京编)` AS brain_region, `手术日期`, `取下时间`
    FROM sample_information_20240815
    """
    sample_result = connection.execute(text(sample_query))
    
    # 使用 DataFrame 构造函数，将查询结果转换为 DataFrame
    sample_df = pd.DataFrame(sample_result.fetchall(), columns=sample_result.keys())
    
    # 只保留 `病人编号` 以 'P' 开头且 `组织编号` 以 'T' 开头的行
    sample_df = sample_df[
        sample_df['病人编号'].str.startswith('P', na=False) &
        sample_df['组织编号'].str.startswith('T', na=False)
    ]
    print('sample_df: ', sample_df)
    
    # 定义函数处理编号，提取数值部分
    def extract_number(s, prefix):
        if pd.isnull(s):
            return None
        s = str(s).strip()
        if s.startswith(prefix):
            num_part = s[len(prefix):]
            # 去掉前导零
            num_part = num_part.lstrip('0')
            if num_part.isdigit():
                return int(num_part)
        return None
    
    # 为 sample_df 添加 P_number_value 和 T_number_value 列，提取数值部分
    sample_df['P_number_value'] = sample_df['病人编号'].apply(lambda x: extract_number(x, 'P'))
    sample_df['T_number_value'] = sample_df['组织编号'].apply(lambda x: extract_number(x, 'T'))
    
    # 为 merged_df 添加 P_number_value 和 T_number_value 列，提取数值部分
    # 修改分隔符为 '_'
    merged_df['P_number_value'] = merged_df['PTRS(B)'].apply(lambda x: extract_number(x.split('-')[0], 'P'))
    merged_df['T_number_value'] = merged_df['PTRS(B)'].apply(
        lambda x: extract_number(x.split('-')[1], 'T') if len(x.split('-')) > 1 else None
    )

    # 使用数值部分进行合并
    extracted_df = pd.merge(merged_df, sample_df, on=['P_number_value', 'T_number_value'], how='left')
    print('extracted_df: ', extracted_df)
    
    # 处理 tissue_dissection_time 列
    def format_tissue_dissection_time(row):
        surgery_date = row['手术日期']
        removal_time = row['取下时间']
        
        # 如果手术日期为空，返回空字符串
        if pd.isnull(surgery_date):
            return ''
        
        # 如果取下时间为空或异常，填充为 '00:00'
        if pd.isnull(removal_time) or removal_time in [None, '', '--', '-']:
            removal_time = '00:00'
        
        try:
            # 清洗 removal_time 字符串
            removal_time = str(removal_time).strip()
            # 使用正则表达式匹配有效的时间格式 HH:MM，分钟部分可选
            match = re.match(r'^(\d{1,2})(?:[:-]?(\d{0,2}))?$', removal_time)
            if match:
                hours = match.group(1).zfill(2)
                minutes = match.group(2).zfill(2) if match.group(2) else '00'
                removal_time_clean = f"{hours}:{minutes}"
                datetime_str = f"{surgery_date} {removal_time_clean}"
                datetime_obj = pd.to_datetime(datetime_str, errors='coerce')
                if pd.isnull(datetime_obj):
                    print(f"日期转换错误: 无法解析日期时间字符串 '{datetime_str}'")
                    return str(surgery_date)
                return datetime_obj.strftime('%Y-%m-%d %H:%M')
            else:
                print(f"无效的时间格式: '{removal_time}'")
                # 无效的时间格式，返回日期部分
                return str(surgery_date)
        except Exception as e:
            print(f"日期转换错误: {e}")
            return str(surgery_date)

    extracted_df['tissue_dissection_time'] = extracted_df.apply(format_tissue_dissection_time, axis=1)
    
    # 重命名 brain_region 列
    extracted_df.rename(columns={'brain_region': 'brain_region'}, inplace=True)
    
    # 删除辅助列
    extracted_df.drop(columns=['P_number_value', 'T_number_value'], inplace=True)
    
    return extracted_df

# 3. 添加新列并生成 cell.csv 文件
def generate_cell_csv(extracted_df,outpath):
    
    # 提取 P、T、R、S 部分（保持不变）
    extracted_df['patient_number'] = extracted_df['PTRS(B)'].str.extract(r'(P\d+)')
    extracted_df['tissue_block_number'] = extracted_df['PTRS(B)'].str.extract(r'(T\d+)')
    extracted_df['small_number'] = extracted_df['PTRS(B)'].str.extract(r'(R\d+)')
    extracted_df['slice_number'] = extracted_df['PTRS(B)'].str.extract(r'(S\d+)')
    
    # 1. 计算 pre_fixation_days
    def compute_pre_fixation_days(row):
        try:
            sample_preparation_date = pd.to_datetime(row['sample_preparation_date'])
            tissue_dissection_time = pd.to_datetime(row['tissue_dissection_time'])
            delta_days = (sample_preparation_date.date() - tissue_dissection_time.date()).days - 1
            if delta_days == -1:
                delta_days = 0
            return delta_days
        except Exception as e:
            return ''

    extracted_df['pre_fixation_days'] = extracted_df.apply(compute_pre_fixation_days, axis=1)

    # 3. 计算 slicing_and_fixation_order
    def compute_slicing_and_fixation_order(row):
        if row['pre_fixation_days'] == 0:
            return 1
        elif row['pre_fixation_days'] != '':
            return 0
        else:
            return ''

    extracted_df['slicing_and_fixation_order'] = extracted_df.apply(compute_slicing_and_fixation_order, axis=1)
    
    # 4. 计算 post_perfusion_4percent_pfa
    def compute_post_perfusion_4percent_pfa(row):
        if row['fresh_perfusion'] == '0':
            return 1
        elif row['fresh_perfusion'] == '1':
            return 0
        else:
            return ''

    extracted_df['post_perfusion_4percent_pfa'] = extracted_df.apply(compute_post_perfusion_4percent_pfa, axis=1)
    
    # 添加其他默认列
    extracted_df['confirmed'] = '--'
    extracted_df['slicing_method'] = 0
    extracted_df['post_perfusion_10percent_formalin'] = 0
    extracted_df['confirmed_0'] = '--'
    extracted_df['image_cell_id'] = '--'
    extracted_df['confirmed_1'] = '--'
    extracted_df['reconstruction_staff'] = '--'
    extracted_df['status'] = '--'
    extracted_df['inspection_staff'] = '--'
    extracted_df['status_0'] = '--'
    extracted_df['sealed_slide'] = '--'
    extracted_df['status_1'] = '--'
    extracted_df['dye_solvent'] = 'DDWater'
    extracted_df['remarks'] = '--'

    extracted_df['perfusion_start_time'] = '--'
    extracted_df['perfusion_end_time'] = '--'
    extracted_df['after_surgery_hours'] = '--'
    extracted_df['cell_depth'] = '--'
    extracted_df['perfusion_current'] = '--'
    extracted_df['perfusion_time_on'] = '--'
    extracted_df['perfusion_time_off'] = '--'	
    extracted_df['inject_method'] = '--'

    extracted_df['experiment_humidity'] = '--'
    extracted_df['perfusion_date'] = '--'
    extracted_df['post_fixation_days'] = '--'  
    extracted_df['dye_concentration'] = '--'

    extracted_df['dye_name'] = '--'
    extracted_df['ihc_category'] = '--'
    extracted_df['immunohistochemistry'] = '--'
    extracted_df['primaryAntibody_concentration'] = '--'
    extracted_df['secondAntibody_band'] = '--'
    extracted_df['DAPI_concentration'] = '--'
    
    # **处理 shooting_date 列，保留日期部分**
    def process_shooting_date(x):
        if pd.isnull(x) or x in ['', '--']:
            return ''
        try:
            date_obj = pd.to_datetime(x)
            return date_obj.strftime('%Y-%m-%d')
        except Exception as e:
            return x  # 如果解析失败，保留原始值

    extracted_df['shooting_date'] = extracted_df['shooting_date'].apply(process_shooting_date)
    # extracted_df['perfusion_date'] = df['perfusion_date'].dt.strftime('%Y-%m-%d')

    # 指定需要的列
    columns = [
        'PTRS(B)', 'injection_ID', 'injection_file', 'metadata_file', 'apo_file', 
        'patient_number', 'tissue_block_number', 'small_number', 'slice_number',
        'confirmed', 'slicing_method', 'slicing_and_fixation_order', 'pre_fixation_days', 'post_fixation_days',
        'sample_preparation_staff', 'sample_preparation_date', 'fresh_perfusion', 'post_perfusion_4percent_pfa', 'post_perfusion_10percent_formalin',
        'brain_region', 'slice_thickness', 'confirmed_0', 'tissue_dissection_time', 'perfusion_start_time', 'perfusion_end_time', 'after_surgery_hours',
        'cell_depth', 'perfusion_current', 'perfusion_time_on', 'perfusion_time_off',
        'dye_name', 'dye_concentration', 'experiment_temperature', 'experiment_humidity',
        'inject_method', 'perfusion_date', 'perfusion_staff', 'ihc_category', 'immunohistochemistry', 'primaryAntibody_concentration', 'secondAntibody_band', 'DAPI_concentration',
        'imaging_device', 'laser_wavelength', 'laser_power', 'laser_power_ratio', 'gain', 'scanner', 'averaging', 'pmt_voltage',
        'z_size', 'tiling', 'overlap', 'xy_resolution', 'z_resolution', 'document_name', 'image_size', 'image_cell_id',
        'shooting_date', 'shooting_staff', 'confirmed_1', 'reconstruction_staff', 'status', 'inspection_staff', 'status_0',
        'sealed_slide', 'status_1', 'dye_solvent', 'remarks', 'soma_x', 'soma_y', 'soma_z', 'imaging_idx',
    ]

    # 确保所有需要的列都存在
    missing_columns = set(columns) - set(extracted_df.columns)
    if missing_columns:
        for col in missing_columns:
            extracted_df[col] = ''

    final_df = extracted_df[columns]

    # 保存为 CSV 文件
    # final_df.to_csv('/mnt/nfs/cell_without_cellID.csv', index=False)
    final_df.to_csv(outpath, index=False, encoding='utf-8')
    print("cell_without_cellID.csv 文件已生成。")
    
    return final_df

# 4. 处理 cell.csv 文件，添加Cell ID和Image Cell ID并修改所有列名
def process_cell_csv(df,start_cell_id, outpath):
    # 删除指定的三列
    df.drop(['injection_file', 'metadata_file', 'apo_file'], axis=1, inplace=True)
    
    # 新增Cell ID列
    df.insert(2, 'Cell ID', range(start_cell_id, start_cell_id + len(df)))

    # # 删除指定的四列
    # df.drop(['injection_ID', 'injection_file', 'metadata_file', 'marker_file'], axis=1, inplace=True)
    
    # 将相同document_name的数据中image_cell_id设为该组数据中Cell ID最小的值
    df['image_cell_id'] = df.groupby('document_name')['Cell ID'].transform('min')

    # 检查列数并调试
    print("当前列数:", len(df.columns))
    print("当前列名:", df.columns)

    # 按照指定的列名依次重命名
    df.columns = [
        'PTRS(B)', 'Injection ID', 'Cell ID', '病人编号', '组织块编号', '小编号', '切片编号', 'confirmed',
        '切片方式(0:振动切片;1:冰冻切片)', '切片与固定顺序(0:先固定;1:先切片)', '切片前固定天数', '切片后固定天数',
        '样本制备人员', '样本制备日期', '新鲜灌注(0:否;1:是)', '4%多聚甲醛固定后灌注(0:否;1:是)',
        '10%甲醛固定后灌注(0:否;1:是)', '脑区', '切片厚度(微米)', 'confirmed_[0]', '组织离体时间',
        '灌注开始时间(e.g.14:30)', '灌注结束时间(e.g.18:30)', 'after surgery(h)', '细胞深度(微米)',
        '灌注电流(nA)', '灌注时间(通电,mins)', '灌注时间(未通电,mins)', '染料名称', '染料浓度(%)',
        '实验温度(℃)', '实验湿度(%RH)', 'Manual/Auto Inject(0:Manual;1:Auto)', '灌注日期',
        '灌注人员', '类别', '免疫染色(0:否;1:是)', '一抗浓度', '二抗波段', 'DAPI浓度', 'Image Device', 
        '激光波长(nm)', '激光功率(mW)', '激光功率比例(%)', 'gain', 'scanner', 'averaging', 'PMT(V)',
        'Z_size(张)', '平铺(0:否;1:是)', 'overlap', 'xy拍摄分辨率(*10e-3μm/px)',
        'z拍摄分辨率(*10e-3μm/px)', 'Document Name', 'Image Size', 'Image Cell ID',
        '拍摄日期', '拍摄人员', 'confirmed_[1]', '重建人员', 'status', '检查人员', 'status_[0]',
        '是否封片', 'status_[1]', '染料溶剂', '备注', 'soma_x', 'soma_y', 'soma_z', 'imaging_idx', 
    ]

    df.to_csv(outpath, index=False)
    print("cell_with_cellID.csv 文件已生成。")


def generate_singleColor_cell_csv(ptrsb,start_cell_id,outpath):
    # 第一步：提取 imaging 和 injection 数据
    merged_df = extract_imaging_and_injection_data(ptrsb)

    # 第二步：提取 sample 信息并合并
    extracted_df = extract_sample_information(merged_df)
    # 第三步：增加其它信息, 并生成 cell.csv 文件
    final_df = generate_cell_csv(extracted_df, outpath)

    # 第四步：处理 cell.csv 文件
    process_cell_csv(final_df, start_cell_id, outpath)

    # # 第三步：增加其它信息, 并生成 cell.csv 文件
    # final_df = generate_cell_csv(extracted_df,'/mnt/nfs/cell_without_cellID.csv')
    #
    # # 第四步：处理 cell.csv 文件
    # process_cell_csv(final_df, start_cell_id, '/mnt/nfs/cell_with_cellID.csv')

if __name__ == "__main__":
    # 第一步：提取 imaging 和 injection 数据
    merged_df = extract_imaging_and_injection_data()
    
    # 第二步：提取 sample 信息并合并
    extracted_df = extract_sample_information(merged_df)
    
    # 第三步：增加其它信息, 并生成 cell.csv 文件
    final_df = generate_cell_csv(extracted_df)

    # 第四步：处理 cell.csv 文件
    process_cell_csv(final_df)
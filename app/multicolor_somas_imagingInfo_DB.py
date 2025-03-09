import pandas as pd
import re
import xml.etree.ElementTree as ET
import openpyxl
import os
import sqlalchemy as db
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# MySQL 连接设置
# DATABASE_URI = 'mysql+pymysql://root:braintell%40seu@localhost/human_neuron'
DATABASE_URI = 'mysql+pymysql://root:hneuronbyseu123@localhost/human_neuron'
# DATABASE_URI = 'mysql+pymysql://root:wlj990521@10.194.35.182/hndb'
engine = db.create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()
metadata = db.MetaData()

# 连接到已存在的 imaging_information 表
imaging_information = db.Table('imaging_information_20241023', metadata, autoload_with=engine)

### 提取 .apo 文件中的信息 ###
def is_float(s):
    pattern = r'^[-+]?[0-9]*\.[0-9]+$'
    return bool(re.match(pattern, s))

def is_integer(s):
    return s.isdigit()

def read_apo(file):
    result = None
    with open(file, 'r', encoding='utf-8') as f:
        result_dict = {}
        headers = []
        data_started = False
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('##'):
                # 处理标题行
                headers_raw = [header.strip() for header in line.strip('#').strip().split(',')]
                headers = []
                for idx, header in enumerate(headers_raw):
                    if header == '':
                        header = f'Unnamed_{idx}'
                    headers.append(header)
                for header in headers:
                    result_dict[header] = []
                data_started = True
                continue
            else:
                if not headers:
                    raise Exception('Headers are missing before data!')
                cells = [cell.strip() for cell in line.strip().split(',')]
                if len(cells) != len(headers):
                    raise Exception(f'行 {line_num} 数据列数与标题列数不一致！预期 {len(headers)} 列，实际 {len(cells)} 列。行内容：{line}')
                for i in range(len(headers)):
                    header = headers[i]
                    cell = cells[i] if i < len(cells) else ''
                    # 转换为适当的类型
                    if is_integer(cell):
                        cell = int(cell)
                    elif is_float(cell):
                        cell = float(cell)
                    result_dict[header].append(cell)
        if not result_dict:
            raise Exception('No data found in file.')
        result = pd.DataFrame(result_dict)
    return result

def filter_apo(data):
    # 只保留 'x', 'y', 'z' 列
    data = data.loc[:, ['x', 'y', 'z']]
    data.columns = ['soma_x', 'soma_y', 'soma_z']
    # 添加 'ID' 列，默认值为 '--'
    data['ID'] = '--'
    # 调整列顺序
    data = data[['ID', 'soma_x', 'soma_y', 'soma_z']]
    return data

### 提取 .xlsx 或 .xml 文件中的成像信息 ###
class ImagingInfo:
    def __init__(self, imaging_path=None):
        self.dye_names = []
        self.info = {}
        if imaging_path is not None:
            self.read_from_file(imaging_path)

    def _initialize_defaults(self):
        self.imaging_device = '' ## 更新
        self.laser_wavelength = '--'
        self.laser_power = '-'  ## 无数据
        self.laser_power_ratio = '--'
        self.gain = '--'
        self.scanner = '--'
        self.averaging = '--'  
        self.pmt_voltage = '-'  ## 无数据
        self.z_size = ''  ## 更新
        self.tiling = '1'     
        self.overlap = '10'   
        self.xy_resolution = ''  ## 更新
        self.z_resolution = ''  ## 更新
        self.document_name = ''  ## 更新
        self.image_size = '--'
        self.shooting_date = ''  ## 更新
        self.shooting_staff = ''  ## 更新

    def read_from_file(self, imaging_path):
        # 初始化默认属性
        self._initialize_defaults()

        # 提取文件名和扩展名
        imaging_file_name = os.path.basename(imaging_path)
        imaging_file, imaging_extension = os.path.splitext(imaging_file_name)

        # 提取 document_name 和 shooting_staff
        parts = imaging_file.split('-')
        if len(parts) >= 5:
            # 除去最后一部分作为 shooting_staff，前面的部分作为 document_name
            self.document_name = '-'.join(parts[:-1])
            # 最后一部分为 shooting_staff
            self.shooting_staff = parts[-1]
        else:
            self.document_name = imaging_file  # 默认值，防止文件名格式不匹配
            self.shooting_staff = 'Unknown'  # 默认值

        # 根据文件扩展名解析文件内容
        if imaging_extension.lower() == '.xml':
            self._parse_xml(imaging_path)
        elif imaging_extension.lower() == '.xlsx':
            self._parse_xlsx(imaging_path)

    def _parse_xml(self, imaging_path):
        # 设置默认设备为 BRUKER
        self.imaging_device = 'BRUKER'
        self.laser_wavelength = 'NaN'
        self.laser_power = 'NaN'
        self.laser_power_ratio = 'NaN'

        self.pixelsPerLine = ''
        self.linesPerFrame = ''
        
        pmt_values = []  # 初始化存储 PMT 电压值的列表
        
        # 解析 XML 文件
        tree = ET.parse(imaging_path)
        root = tree.getroot()
        
        # 提取拍摄日期 (shooting_date)
        if 'date' in root.attrib:
            shooting_date = root.attrib['date'].strip()  # 示例: "7/25/2024 9:29:13 AM"
            if shooting_date:
                tmpStrDate, tmpStrTime, _ = shooting_date.split()
                tmpM, tmpD, tmpY = [int(x) for x in tmpStrDate.split('/')]
                tmpH, tmpMinute, tmpS = [int(x) for x in tmpStrTime.split(':')]
                self.shooting_date = f"{tmpY:04d}-{tmpM:02d}-{tmpD:02d} {tmpH:02d}:{tmpMinute:02d}:{tmpS:02d}"

        # 提取其他成像相关信息
        for obj in root.iter('PVStateValue'):
            if 'key' in obj.attrib:
                key = obj.attrib['key']
                value = obj.attrib.get('value', '')

                # 提取 pixelsPerLine 和 linesPerFrame
                if key == 'pixelsPerLine' and value:
                    self.pixelsPerLine = value
                elif key == 'linesPerFrame' and value:
                    self.linesPerFrame = value
                
                # 提取分辨率信息
                elif key == 'micronsPerPixel':
                    for obj2 in obj.iter('IndexedValue'):
                        if 'index' in obj2.attrib and 'value' in obj2.attrib:
                            index = obj2.attrib['index']
                            if index == 'XAxis':
                                self.xy_resolution = float(obj2.attrib['value']) * 1000  # 转换为纳米
                            elif index == 'ZAxis':
                                self.z_resolution = float(obj2.attrib['value']) * 1000  # 转换为纳米
                
                # 提取 PMT 电压值
                elif key == 'pmtGain':
                    for obj2 in obj.iter('IndexedValue'):
                        if obj2.attrib.get('index') == '1' and 'value' in obj2.attrib:
                            pmt_value = float(obj2.attrib['value'])
                            pmt_values.append(pmt_value)

        # 计算 PMT 电压范围
        if pmt_values:
            min_pmt = min(pmt_values)
            max_pmt = max(pmt_values)
            if min_pmt == max_pmt:
                self.pmt_voltage = f"{min_pmt}"
            else:
                self.pmt_voltage = f"{min_pmt}-{max_pmt}"
        else:
            self.pmt_voltage = ''

        # 提取 Z 轴尺寸 (z_size)
        self.z_size = 0  # 初始值
        for obj in root.iter('Frame'):
            if 'index' in obj.attrib:
                self.z_size = max(self.z_size, int(obj.attrib['index']))

        # 提取重叠 (overlap) 和拼接模式 (tiling)
        for obj in root.iter('Sequence'):
            if 'xYStageGridOverlapPercentage' in obj.attrib:
                self.overlap = obj.attrib['xYStageGridOverlapPercentage']
            if 'type' in obj.attrib:
                self.tiling = '1' if obj.attrib['type'] == 'AtlasVolume' else '0'

        # 计算图像尺寸 (image_size)
        if self.pixelsPerLine and self.linesPerFrame:
            self.image_size = f"{self.linesPerFrame}*{self.pixelsPerLine}"

    def _parse_xlsx(self, imaging_path):
        wb = openpyxl.load_workbook(imaging_path)
        sheet = wb['Image Fields']

        # 默认值初始化
        self.imaging_device = 'NIKON'

        for row in sheet.iter_rows(values_only=True):
            row = [str(cell).strip() if cell is not None else '' for cell in row]
            row_str = " ".join(row).strip()

            # 基于 "Calibration" 字段 解析XY轴分辨率 (XY Resolution) 
            if 'Calibration' in row_str:
                # 使用更通用的正则表达式匹配格式
                calibration_match_result = re.search(r'Calibration.*:\s*([0-9.]+)', row_str)
                print("xy_resolution: ", calibration_match_result)
                if calibration_match_result:
                    try:
                        self.xy_resolution = float(calibration_match_result.group(1)) * 1000  # 转换为纳米
                    except ValueError:
                        # 如果解析的值无法转换为浮点数，则保持默认值或记录日志
                        self.xy_resolution = '-'

            # 解析拍摄日期 (Date)
            elif 'Date' in row[0]:
                if isinstance(row[1], datetime):
                    self.shooting_date = row[1].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    self.shooting_date = row[1]
        
        # 从 "Experiment Data" 表提取z_resolution和z_size
        sheet = wb['Experiment Data']
        tag = False
        z_resolution_pos = -1
        z_size_pos = -1

        for row in sheet.iter_rows(values_only=True):
            row = [str(cell).strip() if cell is not None else '' for cell in row]
            if 'Count' in row:
                tag = True
                for i in range(len(row)):
                    if 'Z-Step' in row[i]:
                        z_resolution_pos = i
                    elif 'Count' in row[i]:
                        z_size_pos = i
                continue
            if tag:
                if z_resolution_pos >= 0 and row[z_resolution_pos]:
                    self.z_resolution = float(row[z_resolution_pos]) * 1000  # 转换为纳米
                if z_size_pos >= 0 and row[z_size_pos]:
                    self.z_size = row[z_size_pos]     
                tag = False

def insert_to_db(apo_data, imaging_info, apo_file, imaging_file):
    # 检查数据库中是否已经存在该组合的 apo_file 和 metadata_file
    apo_file_basename = os.path.basename(apo_file)
    imaging_file_basename = os.path.basename(imaging_file)

    # 如果已处理过，直接返回
    if '_DB' in apo_file_basename or '_DB' in imaging_file_basename:
        print(f"文件已处理过，跳过: {apo_file_basename} 和 {imaging_file_basename}")
        return

    # 插入数据
    for index, apo in apo_data.iterrows():
        data_row = {
            'ID': apo['ID'],  # ID 默认为 '--'
            'imaging_device': imaging_info.imaging_device,
            'laser_wavelength': imaging_info.laser_wavelength,
            'laser_power': imaging_info.laser_power,
            'laser_power_ratio': imaging_info.laser_power_ratio,
            'gain': imaging_info.gain,
            'scanner': imaging_info.scanner,
            'averaging': imaging_info.averaging,
            'pmt_voltage': imaging_info.pmt_voltage,
            'z_size': imaging_info.z_size,
            'tiling': imaging_info.tiling,
            'overlap': imaging_info.overlap,
            'xy_resolution': imaging_info.xy_resolution,
            'z_resolution': imaging_info.z_resolution,
            'document_name': imaging_info.document_name,
            'image_size': imaging_info.image_size, 
            'shooting_date': imaging_info.shooting_date,
            'shooting_staff': imaging_info.shooting_staff,
            'soma_x': apo['soma_x'],
            'soma_y': apo['soma_y'],
            'soma_z': apo['soma_z'],
            'apo_file': apo_file_basename,    # 更新字段名
            'metadata_file': imaging_file_basename
        }
        stmt = imaging_information.insert().values(data_row)
        session.execute(stmt)
    session.commit()
    print(f"成功插入数据：{apo_file_basename} 和 {imaging_file_basename}")

    # 重命名已处理的文件，添加 '_DB' 后缀
    # try:
    #     apo_file_root, apo_file_ext = os.path.splitext(apo_file)
    #     new_apo_file = apo_file_root + '_DB' + apo_file_ext
    #     os.rename(apo_file, new_apo_file)
    #     print(f"重命名 apo 文件：{apo_file} -> {new_apo_file}")
    #
    #     imaging_file_root, imaging_file_ext = os.path.splitext(imaging_file)
    #     new_imaging_file = imaging_file_root + '_DB' + imaging_file_ext
    #     os.rename(imaging_file, new_imaging_file)
    #     print(f"重命名元数据文件：{imaging_file} -> {new_imaging_file}")
    # except Exception as e:
    #     print(f"重命名文件时出错：{e}")

def extract_identifier(filename):
    """
    从文件名中提取用于匹配的编号部分，例如：
    输入：'P00095-T001-R001-S026-1.apo'
    输出：'P00095-T001-R001-S026-1'
    """
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    return name  # 返回文件名，不含扩展名


def process_single_file_pair(apo_file_path, metadata_file_path):
    """
    Process a single apo file and metadata file pair

    Parameters:
        apo_file_path: Path to the .apo file
        metadata_file_path: Path to the metadata file (.xlsx or .xml)

    Returns:
        Dictionary with status and processing information
    """
    try:
        # Check if files exist
        if not os.path.exists(apo_file_path) or not os.path.exists(metadata_file_path):
            missing = []
            if not os.path.exists(apo_file_path): missing.append("APO文件")
            if not os.path.exists(metadata_file_path): missing.append("元数据文件")
            return {"status": "error", "message": f"文件不存在: {', '.join(missing)}"}

        # Get file basenames
        apo_file_basename = os.path.basename(apo_file_path)
        metadata_file_basename = os.path.basename(metadata_file_path)

        # Check if already processed
        if '_DB' in apo_file_basename or '_DB' in metadata_file_basename:
            return {"status": "skipped", "message": f"文件已处理过: {apo_file_basename} 和 {metadata_file_basename}"}

        # Read and process files
        apo_data = read_apo(apo_file_path)
        apo_data = filter_apo(apo_data)

        # Read metadata
        imaging_info = ImagingInfo(metadata_file_path)

        # Insert data to database
        insert_to_db(apo_data, imaging_info, apo_file_path, metadata_file_path)

        return {
            "status": "success",
            "message": f"成功处理: {apo_file_basename} 和 {metadata_file_basename}",
            "apo_file": apo_file_basename,
            "metadata_file": metadata_file_basename
        }
    except Exception as e:
        return {"status": "error", "message": f"处理错误: {str(e)}"}
# 保留原批量处理功能但改为调用单文件处理函数
if __name__ == "__main__":
    # apo_dir = 'C:\\Users\\kaixiang\\Downloads\\Multicolor_Somas'
    # metadata_dir = 'C:\\Users\\kaixiang\\Downloads\\Metadata'

    apo_dir = '/mnt/nfs/hndb/Imaging_Files/Multicolor_Somas'
    metadata_dir = '/mnt/nfs/hndb/Imaging_Files/Metadata'

    # 获取所有的 .apo 文件，排除已处理的文件
    apo_files = [os.path.join(apo_dir, f) for f in os.listdir(apo_dir) 
                    if f.endswith('.apo') and '_DB' not in f]

    for apo_file in apo_files:
        # 提取标记文件的编号（不包含扩展名）
        identifier = extract_identifier(apo_file)

        # 在元数据目录中查找以 identifier 开头，紧跟 '-'，然后是任意字符的 .xml 或 .xlsx 文件，排除已处理的文件
        matching_files = []
        for f in os.listdir(metadata_dir):
            if '_DB' in f:
                continue  # 跳过已处理的文件
            metadata_name, metadata_ext = os.path.splitext(f)
            if metadata_ext.lower() in ['.xml', '.xlsx']:
                # 检查文件名是否以标记文件名为前缀，且紧跟一个 '-'
                if metadata_name.startswith(identifier + '-'):
                    matching_files.append(os.path.join(metadata_dir, f))

        if not matching_files:
            print(f"未找到匹配的元数据文件：{identifier}")
            continue

        if len(matching_files) > 1:
            print(f"找到多个匹配的元数据文件，跳过：{identifier}")
            continue

        # 如果有且仅有一个匹配的元数据文件，进行处理
        imaging_file = matching_files[0]

        # 检查是否已经处理过
        apo_file_basename = os.path.basename(apo_file)
        imaging_file_basename = os.path.basename(imaging_file)

        if '_DB' in apo_file_basename or '_DB' in imaging_file_basename:
            print(f"文件已处理过，跳过: {apo_file_basename} 和 {imaging_file_basename}")
            continue

        try:
            # 读取 apo 文件数据
            apo_data = read_apo(apo_file)
            apo_data = filter_apo(apo_data)

            # 读取 imaging 文件数据
            imaging_info = ImagingInfo(imaging_file)

            # 插入数据到数据库，传入 apo_file 和 imaging_file
            insert_to_db(apo_data, imaging_info, apo_file, imaging_file)
        except Exception as e:
            print(f"处理文件时出错：{apo_file_basename}, 错误信息：{e}")
            continue
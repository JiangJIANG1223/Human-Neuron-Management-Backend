import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, insert
from sqlalchemy.orm import sessionmaker
import numpy as np
import os

# 1. 读取 CSV 文件
# csv_file = 'Cell_40810_44661_singleColor.csv'
csv_file = '/mnt/nfs/batch_import_cellINFO/Cell_62399_62559_multicolor.csv'
df = pd.read_csv(csv_file, encoding='UTF-8')

# 2. 对每个 Document Name 分组
for doc_name, group_df in df.groupby('Document Name'):
    # 3. 构造一个新的 DataFrame，对应 Marker 文件字段
    out_df = pd.DataFrame()

    # 对坐标列做 float 转换
    out_df['##x'] = group_df['soma_x'].astype(float)
    out_df['y']   = group_df['soma_y'].astype(float)
    out_df['z']   = group_df['soma_z'].astype(float)

    # 固定值：radius=1, shape=1, 颜色=红(255,0,0)
    out_df['radius'] = 1
    out_df['shape']  = 1
    out_df['color_r'] = 255
    out_df['color_g'] = 0
    out_df['color_b'] = 0

    # name 列取自原始数据的 'Cell ID'
    out_df['name'] = group_df['Cell ID'].astype(str)

    # comment 列取自分组名 doc_name（即 Document Name）
    out_df['comment'] = doc_name

    # 4. 指定 Marker 文件的输出文件夹和文件名
    marker_folder = "/mnt/nfs/hndb/Imaging_Files/PTRSB_Somas"
    # marker_folder = "PTRSB_Somas"

    # 指定输出文件名：<document_name>_somalist.marker
    out_filename = f"{doc_name}_somalist.marker"

    # 将目录和文件名拼接为完整输出路径
    out_path = os.path.join(marker_folder, out_filename)

    # 5. 指定列顺序，导出 Marker 文件
    col_order = [
        '##x','y','z','radius','shape','name','comment',
        'color_r','color_g','color_b'
    ]
    out_df[col_order].to_csv(
        out_path,
        index=False,
        float_format='%.3f' 
    )

    print(f"已生成 {out_filename} ，包含 {len(group_df)} 个细胞。")

# 6. 将数据去除插入到 human_neuron 数据库 human_singlecell_trackingtable_20240712表中
df['soma_x'] = '--'
df['soma_y'] = '--'
df['soma_z'] = '--'
df = df.replace({np.nan: '--'})

# engine = create_engine("mysql+pymysql://root:braintell%40seu@localhost/human_neuron")
engine = create_engine("mysql+pymysql://root:hneuronbyseu123@localhost/human_neuron")
metadata = MetaData()

table_name = 'human_singlecell_trackingtable_20240712'
tracking_table = Table(table_name, metadata, autoload_with=engine)

Session = sessionmaker(bind=engine)
session = Session()

try:
    for index, row in df.iterrows():
        stmt = insert(tracking_table).values(row.to_dict())

        session.execute(stmt)

    session.commit()
    print("数据已成功插入到human_singlecell_trackingtable_20240712表中")

except Exception as e:
    
    print(f"数据插入失败: {e}")
    session.rollback()

finally:
    session.close()

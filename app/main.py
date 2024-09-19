# FastAPI应用的入口，包含路由定义和数据库连接配置
import shutil
import cv2
import imageio
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi.middleware.cors import CORSMiddleware
from . import models, schemas, crud
from .database import SessionLocal
from fastapi.staticfiles import StaticFiles
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from PIL import Image
import os
from app.models import User, DailyReport
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
from sqlalchemy import desc
import uvicorn
import json
from collections import OrderedDict
import zipfile
from sqlalchemy import asc
import aiofiles
import shutil
from collections import OrderedDict
import zipfile
import sys
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table, update, select
from sqlalchemy.orm import sessionmaker
import pandas as pd
import matplotlib.pyplot as plt
import logging


app = FastAPI()
# 23333
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 MIP_Downsample 目录
# app.mount("/MIP_Downsample", StaticFiles(directory="./MIP_Downsample"), name="MIP_Downsample")

# 挂载 static 目录，服务静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database connection URL
#DATABASE_URL = "mysql+pymysql://root:braintell%40seu@localhost/Human_Neuron"
DATABASE_URL = "mysql+pymysql://root:hneuronbyseu123@localhost/human_neuron"

# Dependency to get the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Exception handler for AuthJWTException
@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message}
    )

@AuthJWT.load_config
def get_config():
    return schemas.Settings()

'''****************************SWC可视化开始***************************'''
def readSWC(swc_path, mode='simple'): # pandas DataFrame
    n_skip = 0
    with open(swc_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break
    # names = ["##n", "type", "x", "y", "z", "r", "parent", "seg_id", "level", "mode", "timestamp", "teraflyindex"]
    names = ["##n", "type", "x", "y", "z", "r", "parent"]
    used_cols = [0, 1, 2, 3, 4, 5, 6]
    if mode == 'simple':
        pass
    df = pd.read_csv(swc_path, index_col=0, skiprows=n_skip, sep=" ",
                     usecols=used_cols,
                     names=names
                     )
    # 检查 ##n 列中是否有两个 1
    indices = df.index[df["parent"] == -1].tolist()
    if len(indices) == 2:
        # 如果有两个 1，则只保留第二个 1 之前的内容
        df = df.iloc[:len(df) // 2]

    return df

def get_degree(tswc):   # Degree of node: the number of nodes connected to it
    tswc['degree'] = tswc['parent'].isin(tswc.index).astype('int')
    # print(tswc['degree'])
    n_child = tswc.parent.value_counts()
    n_child = n_child[n_child.index.isin(tswc.index)]
    tswc.loc[n_child.index, 'degree'] = tswc.loc[n_child.index, 'degree'] + n_child
    return tswc

def get_rid(swc):
    '''
    Find root node.
    '''
    rnode=swc[((swc['parent']<0) & (swc['type']<=1))]
    if rnode.shape[0]<1:
        return -1
    return rnode.index[0]

def get_keypoint(swc, rid=None):  # keypoint: degree ≠ 2 (branches & tips)
    if rid is None:
        rid = get_rid(swc)
    # print(swc.shape)
    swc=get_degree(swc)
    idlist = swc[((swc.degree!=2) | (swc.index==rid))].index.tolist()
    return idlist

def swc2branches(swc):
    '''
    reture branch list of a swc
    '''
    keyids=get_keypoint(swc)
    branches=[]
    for key in keyids:
        if (swc.loc[key,'parent']<0) | (swc.loc[key,'type']<=1):
            continue
        branch=[]
        branch.append(key)
        pkey=swc.loc[key,'parent']
        while True:
            branch.append(pkey)
            if pkey in keyids:
                break
            key=pkey
            if (swc.loc[key,'parent']<0):
                break
            pkey=swc.loc[key,'parent']
        branches.append(branch)
    return branches

def show_neuron_swc(swc_path: str):

    """
    Displays a neuron structure from an SWC file.

    Args:
        swc_path (str): The path to the SWC file.

    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
    """
    # Read the SWC file
    swc = readSWC(swc_path)

    # Get the branches of the neuron structure
    branches = swc2branches(swc)
    if x_size != -1 and y_size != -1:
        # Create a new figure and axes
        _, ax = plt.subplots(figsize=(x_size/100, y_size/100))
    else:
        _, ax = plt.subplots(figsize=(5.12, 5.12))
    # Plot the neuron structure
    colors = ['white', 'black', 'red', 'blue', 'magenta', 'green']
    for branch in branches:
        br_color = colors[3]
        #br_color = colors[int(swc.loc[branch[0], 'type']) % len(colors)]
        x_coords = [swc.loc[node, 'x'] for node in branch]
        y_coords = [swc.loc[node, 'y'] for node in branch]
        ax.plot(x_coords, y_coords, color=br_color)

    # Set equal aspect ratio and remove axes
    ax.set_aspect('equal')
    ax.axis('off')
    res=swc_path.replace(".swc",".jpg")
    plt.savefig(res)
    # Close the figure to free up memory
    plt.close()

    # # 调用vaa3d显示SWC
    # vaa3d_exe = 'C://Users//kaixiang//Desktop//Vaa3D-x.1.1.2_Windows_64bit//Vaa3D-x.exe'
    # subprocess.Popen([vaa3d_exe, swc_path])

    return res

def get_swc(swc_file,image_path, cellid, projection_direction='xy', ignore_background=False):

    rawID='-'
    soma_x=0
    soma_y=0
    soma_z=0
    # ▒~Z▒~I▒~U▒▒~M▒▒~SURL
    SQLALCHEMY_DATABASE_URLg = "mysql+pymysql://root:hneuronbyseu123@localhost/human_neuron"
    # ▒~H~[建▒~U▒▒~M▒▒~S▒~U▒~S~N
    engineg = create_engine(SQLALCHEMY_DATABASE_URLg)
    # ▒~H~[建MetaData对象
    metadatag = MetaData()
    # ▒~O~M▒~D▒~[▒▒| ~G表
    tracking_table = Table('human_singlecell_trackingtable_20240712', metadatag, autoload_with=engineg)
    # ▒~H~[建▒~_▒询
    query = select(
        tracking_table.c['Image Cell ID'],
        tracking_table.c.soma_x,
        tracking_table.c.soma_y,
        tracking_table.c.soma_z
    ).where(tracking_table.c['Cell ID'] == cellid)

    # ▒~I▒▒~L▒~_▒询
    with engineg.connect() as connection:
        result = connection.execute(query)

        # ▒~N▒▒~O~V▒~_▒询▒~S▒~^~\
        for row in result:
            rawID=row[0]
            soma_x=row.soma_x
            soma_y=row.soma_y
            soma_z=row.soma_z

    background = imageio.v2.imread(image_path)
    y_size, x_size = background.shape[:2]  # 取前两维，忽略通道数

    if rawID == "-":
        if projection_direction == 'xy':
            projection_axes = 0
        elif projection_direction == 'xz':
            projection_axes = 1
        elif projection_direction == 'yz':
            projection_axes = 2
        else:
            raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")
        # 创建一个全白图像
        background = np.ones((y_size, x_size, 3), dtype=np.uint8) * 255  # 3表示RGB通道

        print(y_size,x_size)
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if(ignore_background):
            background = np.ones_like(background) * 255
        if background.ndim == 2:  # 如果是单通道
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        point_l = Readswc_v2(swc_file)


        # 定义颜色列表
        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0)]

        thickness = 2
        if len(point_l.p) > 1:
            print(int(point_l.p[1].si))
            color = colors[3]
        else:
            # 处理点数量不足的情况，例如使用默认颜色
            color = (0, 0, 255)  # 定义一个默认颜色


        if (projection_axes == 0):
            cv2.circle(background, (int(point_l.p[1].x), int(point_l.p[1].y)), 3, color, -1)
        elif (projection_axes == 1):
            cv2.circle(background, (int(point_l.p[1].x), int(point_l.p[1].z)), 3, color, -1)
        elif (projection_axes == 2):
            cv2.circle(background, (int(point_l.p[1].y), int(point_l.p[1].z)), 3, color, -1)

        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            if (p.p == 0 or p.p == -1): continue
            x, y, z, si = p.x, p.y, p.z, p.si
            px, py, pz, psi = point_l.p[p.p].x, point_l.p[p.p].y, point_l.p[p.p].z, point_l.p[p.p].si
            # y, py = background.shape[1] - y, background.shape[1] - py

            x, y, z, si = int(x), int(y), int(z), int(si)
            px, py, pz, psi = int(px), int(py), int(pz), int(psi)


            if (projection_axes == 0):
                # draw a line between two points
                cv2.line(background, (x, y), (px, py), colors[3], thickness)  #colors-si  -- colors[2]
            elif (projection_axes == 1):
                cv2.line(background, (x, z), (px, pz), colors[3], thickness)
            elif (projection_axes == 2):
                cv2.line(background, (y, z), (py, pz), colors[3], thickness)
        if x_size != 512 or y_size != 512:
            background = cv2.resize(background, (x_size, y_size))

        res=swc_file.replace(".swc",".jpg")
        finalImage=Image.fromarray(background,'RGB')
        finalImage = finalImage.rotate(-180)
        # 进行水平镜像
        mirroredImage = finalImage.transpose(method=Image.FLIP_LEFT_RIGHT)
        mirroredImage.save(res)
        # Close the figure to free up memory


        # # 调用vaa3d显示SWC
        # vaa3d_exe = 'C://Users//kaixiang//Desktop//Vaa3D-x.1.1.2_Windows_64bit//Vaa3D-x.exe'
        # subprocess.Popen([vaa3d_exe, swc_path])

        return res
    else:
        if projection_direction == 'xy':
            projection_axes = 0
        elif projection_direction == 'xz':
            projection_axes = 1
        elif projection_direction == 'yz':
            projection_axes = 2
        else:
            raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")
        # 创建一个全白图像
        background = np.ones((y_size, x_size, 3), dtype=np.uint8) * 255  # 3表示RGB通道

        print(y_size,x_size)
        if(soma_x!='-' and soma_y!='-'):
            x_start = max(int(soma_x) - 256, 0)  # 886 - 256 = 630
            y_start = max(int(soma_y) - 256, 0)  # 800 - 256 = 544
        else:
            x_start=0
            y_start=0
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if(ignore_background):
            background = np.ones_like(background) * 255
        if background.ndim == 2:  # 如果是单通道
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        point_l = Readswc_v2(swc_file)


        # 定义颜色列表
        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0)]

        thickness = 2
        if len(point_l.p) > 1:
            print(int(point_l.p[1].si))
            color = colors[3]
        else:
            # 处理点数量不足的情况，例如使用默认颜色
            color = (0, 0, 255)  # 定义一个默认颜色


        if (projection_axes == 0):
            cv2.circle(background, (int(point_l.p[1].x-x_start), int(point_l.p[1].y-y_start)), 3, color, -1)
        elif (projection_axes == 1):
            cv2.circle(background, (int(point_l.p[1].x), int(point_l.p[1].z)), 3, color, -1)
        elif (projection_axes == 2):
            cv2.circle(background, (int(point_l.p[1].y), int(point_l.p[1].z)), 3, color, -1)

        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            if (p.p == 0 or p.p == -1): continue
            x, y, z, si = p.x, p.y, p.z, p.si
            px, py, pz, psi = point_l.p[p.p].x, point_l.p[p.p].y, point_l.p[p.p].z, point_l.p[p.p].si
            # y, py = background.shape[1] - y, background.shape[1] - py

            x, y, z, si = int(x), int(y), int(z), int(si)
            px, py, pz, psi = int(px), int(py), int(pz), int(psi)


            if (projection_axes == 0):
                # draw a line between two points
                cv2.line(background, (int(x-x_start), int(y-y_start)), (int(px-x_start), int(py-y_start)), colors[3], thickness)  #colors-si  -- colors[2]
            elif (projection_axes == 1):
                cv2.line(background, (x, z), (px, pz), colors[3], thickness)
            elif (projection_axes == 2):
                cv2.line(background, (y, z), (py, pz), colors[3], thickness)
        if x_size != 512 or y_size != 512:
            background = cv2.resize(background, (x_size, y_size))

        res=swc_file.replace(".swc",".jpg")
        finalImage=Image.fromarray(background,'RGB')
        finalImage = finalImage.rotate(-180)
        # 进行水平镜像
        mirroredImage = finalImage.transpose(method=Image.FLIP_LEFT_RIGHT)
        mirroredImage.save(res)
        # Close the figure to free up memory
        # # 调用vaa3d显示SWC
        # vaa3d_exe = 'C://Users//kaixiang//Desktop//Vaa3D-x.1.1.2_Windows_64bit//Vaa3D-x.exe'
        # subprocess.Popen([vaa3d_exe, swc_path])
        return res


class SWCfilepath(BaseModel):
    ss: str  # 定义接收的字段
    mipforswc: str  # 定义接收的字段
    cellid: str

@app.post("/api/getSWC/")
def get_swcimage(request: SWCfilepath):
    globalpath1="/mnt/nfs/hndb"
    repath=globalpath1+request.ss
    mippath=globalpath1+"/"+request.mipforswc
    swcimage=get_swc(repath,mippath,request.cellid)
    swcimagename=os.path.basename(swcimage).split("_")[0]
    try:
        return FileResponse(swcimage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # # 定义数据库URL
    # SQLALCHEMY_DATABASE_URLg = "mysql+pymysql://root:root@localhost/human_neuron"
    #
    # # 创建数据库引擎
    # engineg = create_engine(SQLALCHEMY_DATABASE_URLg)
    #
    # # 创建MetaData对象
    # metadatag = MetaData()
    #
    # # 反射目标表
    # tracking_table = Table('human_singlecell_trackingtable_20240712', metadatag, autoload_with=engineg)
    #
    # # 创建会话
    # Session = sessionmaker(bind=engineg)
    # session = Session()
    # stmt = (
    #     update(tracking_table)
    #         .where(tracking_table.c['Cell ID'] == swcimagename)  # 这里假设表中有一个主键id列
    #         .values(swc_image=swcimage)
    # )
    # session.execute(stmt)
    # # 提交事务
    # session.commit()
    #
    # # 关闭会话
    # session.close()
    #
'''***************************************************SWC可视化结束***************************************************'''
'''***************************************MIP和SWC重叠开始*************************************'''
class swcPoint:
    def __init__(self, sample_number, structure_identifier,
                 x_position, y_position, z_position, radius, parent_sample):
        self.n = sample_number
        # self.si = 0#structure_identifier
        self.si = structure_identifier
        self.x = x_position
        self.y = y_position
        self.z = z_position
        self.r = radius
        self.p = parent_sample
        self.s = [] # sons
        self.fn = -1 # fiber number
        self.conn = [] # connect points in other fiber
        self.mp = [] # match point in other swc
        self.neighbor = [] # neighbor closer than a distance. store neighbor number and connect info. as [d, bool]
        # self.isend = False
        self.ishead = False
        self.istail = False
        self.swcNeig = [] # neighbor closer than a distance.
        self.swcMatchP = []
        self.i = 0
        self.visited = 0
        self.pruned = False
        self.depth = 0

class swcP_list:
    def __init__(self):
        self.p = []
        self.count = 0
def Readswc_v2(swc_name):
    point_l = swcP_list()
    with open(swc_name, 'r' ) as f:
        lines = f.readlines()

    swcPoint_number = -1
    # swcPoint_list = []
    point_list = []
    list_map = np.zeros(500000)

    # 计数器，用于记录 point[6] == -1 的数量
    count_negative_one = 0

    for line in lines:
        if(line[0] == '#'):
            continue
        # 检查 point[6] 是否为 -1

        temp_line = line.split()
        # print(temp_line)
        if int(temp_line[6]) == -1:
            count_negative_one += 1


        # 如果出现了两次 point[6] == -1，则只保留前一半内容
        if count_negative_one == 2:
            break
        point_list.append(temp_line)

        swcPoint_number = swcPoint_number + 1
        list_map[int(temp_line[0])] = swcPoint_number


    swcPoint_number = 0
    for point in point_list:
        swcPoint_number = swcPoint_number + 1
        point[0] = swcPoint_number # int(point[0])
        point[1] = int(point[1])
        point[2] = float(point[2])
        point[3] = float(point[3])
        point[4] = float(point[4])
        point[5] = float(point[5])
        point[6] = int(point[6])
        if(point[6] == -1):
            pass
        else:
            point[6] = int(list_map[int(point[6])]) + 1

    # swcPoint_list.append(swcPoint(0,0,0,0,0,0,0)) # an empty point numbered 0
    point_l.p.append(swcPoint(0,0,0,0,0,0,0))

    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        point_l.p.append(temp_swcPoint)
    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        if not temp_swcPoint.p == -1:
            # parent = swcPoint_list[int(temp_swcPoint.p)]
            parent = point_l.p[int(temp_swcPoint.p)]
            parent.s.append(temp_swcPoint.n)
        if(point[0] == 1):
            point_l.p[int(point[0])].depth = 0
        else:
            point_l.p[int(point[0])].depth = parent.depth + 1
        # point_l.p.append(temp_swcPoint)
    # for i in range(1, 10):
    #     print(point_l.p[i].s)

    return point_l # (swcPoint_list)

def get_mip_swc(swc_file, image, cellid, projection_direction='xy', ignore_background=False):

    rawID='-'
    soma_x=0
    soma_y=0
    soma_z=0

    # ▒~Z▒~I▒~U▒▒~M▒▒~SURL
    SQLALCHEMY_DATABASE_URLg = "mysql+pymysql://root:hneuronbyseu123@localhost/human_neuron"
    # ▒~H~[建▒~U▒▒~M▒▒~S▒~U▒~S~N
    engineg = create_engine(SQLALCHEMY_DATABASE_URLg)
    # ▒~H~[建MetaData对象
    metadatag = MetaData()
    # ▒~O~M▒~D▒~[▒▒| ~G表
    tracking_table = Table('human_singlecell_trackingtable_20240712', metadatag, autoload_with=engineg)
    # ▒~H~[建▒~_▒询
    query = select(
        tracking_table.c['Image Cell ID'],
        tracking_table.c.soma_x,
        tracking_table.c.soma_y,
        tracking_table.c.soma_z
    ).where(tracking_table.c['Cell ID'] == cellid)

    # ▒~I▒▒~L▒~_▒询
    with engineg.connect() as connection:
        result = connection.execute(query)

        # ▒~N▒▒~O~V▒~_▒询▒~S▒~^~\
        for row in result:
            rawID=row[0]
            soma_x=row.soma_x
            soma_y=row.soma_y
            soma_z=row.soma_z
    if rawID=='-':
        if projection_direction == 'xy':
            projection_axes = 0
        elif projection_direction == 'xz':
            projection_axes = 1
        elif projection_direction == 'yz':
            projection_axes = 2
        else:
            raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")
        image_path = image # 注意使用原始字符串，避免转义字符问题
        background = imageio.v2.imread(image_path)
        y_size, x_size = background.shape[:2]  # 取前两维，忽略通道数
        print(y_size,x_size)
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if(ignore_background):
            background = np.ones_like(background) * 255
        if background.ndim == 2:  # 如果是单通道
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        point_l = Readswc_v2(swc_file)


        # 定义颜色列表
        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0)]

        thickness = 2
        if len(point_l.p) > 1:
            print(int(point_l.p[1].si))
            color = colors[3]
        else:
            # 处理点数量不足的情况，例如使用默认颜色
            color = (0, 0, 255)  # 定义一个默认颜色


        if (projection_axes == 0):
            cv2.circle(background, (int(point_l.p[1].x), int(point_l.p[1].y)), 3, color, -1)
        elif (projection_axes == 1):
            cv2.circle(background, (int(point_l.p[1].x), int(point_l.p[1].z)), 3, color, -1)
        elif (projection_axes == 2):
            cv2.circle(background, (int(point_l.p[1].y), int(point_l.p[1].z)), 3, color, -1)

        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            if (p.p == 0 or p.p == -1): continue
            x, y, z, si = p.x, p.y, p.z, p.si
            px, py, pz, psi = point_l.p[p.p].x, point_l.p[p.p].y, point_l.p[p.p].z, point_l.p[p.p].si
            # y, py = background.shape[1] - y, background.shape[1] - py

            x, y, z, si = int(x), int(y), int(z), int(si)
            px, py, pz, psi = int(px), int(py), int(pz), int(psi)


            if (projection_axes == 0):
                # draw a line between two points
                cv2.line(background, (x, y), (px, py), colors[3], thickness)  #colors-si  -- colors[2]
            elif (projection_axes == 1):
                cv2.line(background, (x, z), (px, pz), colors[3], thickness)
            elif (projection_axes == 2):
                cv2.line(background, (y, z), (py, pz), colors[3], thickness)
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (x_size, y_size))

        return background

    else:
        if projection_direction == 'xy':
            projection_axes = 0
        elif projection_direction == 'xz':
            projection_axes = 1
        elif projection_direction == 'yz':
            projection_axes = 2
        else:
            raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")
        image_path = image # 注意使用原始字符串，避免转义字符问题
        background = imageio.v2.imread(image_path)
        y_size, x_size = background.shape[:2]  # 取前两维，忽略通道数
        print(y_size,x_size)

        if(soma_x!='-' and soma_y!='-'):
            x_start = max(int(soma_x) - 256, 0)  # 886 - 256 = 630
            y_start = max(int(soma_y) - 256, 0)  # 800 - 256 = 544
        else:
            x_start=0
            y_start=0


        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if(ignore_background):
            background = np.ones_like(background) * 255
        if background.ndim == 2:  # 如果是单通道
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        point_l = Readswc_v2(swc_file)


        # 定义颜色列表
        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0)]

        thickness = 2
        if len(point_l.p) > 1:
            print(int(point_l.p[1].si))
            color = colors[3]
        else:
            # 处理点数量不足的情况，例如使用默认颜色
            color = (0, 0, 255)  # 定义一个默认颜色


        if (projection_axes == 0):
            cv2.circle(background, (int(point_l.p[1].x-x_start), int(point_l.p[1].y-y_start)), 3, color, -1)
        elif (projection_axes == 1):
            cv2.circle(background, (int(point_l.p[1].x-x_start), int(point_l.p[1].z)), 3, color, -1)
        elif (projection_axes == 2):
            cv2.circle(background, (int(point_l.p[1].y), int(point_l.p[1].z)), 3, color, -1)

        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            if (p.p == 0 or p.p == -1): continue
            x, y, z, si = p.x, p.y, p.z, p.si
            px, py, pz, psi = point_l.p[p.p].x, point_l.p[p.p].y, point_l.p[p.p].z, point_l.p[p.p].si
            # y, py = background.shape[1] - y, background.shape[1] - py

            x, y, z, si = int(x), int(y), int(z), int(si)
            px, py, pz, psi = int(px), int(py), int(pz), int(psi)


            if (projection_axes == 0):
                # draw a line between two points
                cv2.line(background, (int(x-x_start), int(y-y_start)), (int(px-x_start), int(py-y_start)), colors[3], thickness)  #colors-si  -- colors[2]
            elif (projection_axes == 1):
                cv2.line(background, (x, z), (px, pz), colors[3], thickness)
            elif (projection_axes == 2):
                cv2.line(background, (y, z), (py, pz), colors[3], thickness)
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (x_size, y_size))

        return background

class MIP_SWCfilepath(BaseModel):
    image_file: str
    swc_file: str
    cellid: str

@app.post("/api/getMIPSWC/")
def get_mipswc_image(request: MIP_SWCfilepath):
    globalpath="/mnt/nfs/hndb"
    swc=globalpath+request.swc_file
    mip=globalpath+"/"+request.image_file
    # print(swc)
    #print(mip)
    re=get_mip_swc(swc,mip,request.cellid)
    finalImage=Image.fromarray(re,'RGB')
    # 顺时针旋转 180 度
    finalImage = finalImage.rotate(-180)
    # 进行水平镜像
    mirroredImage = finalImage.transpose(method=Image.FLIP_LEFT_RIGHT)
    savepath=os.path.dirname(swc)+"/"+os.path.basename(swc).split("_")[0]+"_Combine"+".jpg"
    mirroredImage.save(savepath)

    try:
        return FileResponse(savepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''*******************************************MIP和SWC重叠结束*******************************'''


'''**************************************单个文件数据转换**************************************'''
def find_storage_path(base_folder, filename):
    file_number = int(filename.split('_')[0])  # 例如 "00028.txt" -> 28

    # 遍历基础文件夹，寻找合适的外层文件夹
    for outer_folder in os.listdir(base_folder):
        outer_folder_path = os.path.join(base_folder, outer_folder)
        if os.path.isdir(outer_folder_path):
            # 获取外层文件夹的范围
            outer_start, outer_end = map(int, outer_folder.split('_'))
            if outer_start <= file_number <= outer_end:
                # 遍历找到的外层文件夹，寻找合适的内层文件夹
                for inner_folder in os.listdir(outer_folder_path):
                    # print(outer_folder_path.split("\\")[-1])
                    f1=outer_folder_path.split("\\")[-1]
                    inner_folder_path = os.path.join(outer_folder_path, inner_folder)
                    if os.path.isdir(inner_folder_path):
                        # 获取内层文件夹的范围
                        inner_start, inner_end = map(int, inner_folder.split('_'))
                        if inner_start <= file_number <= inner_end:
                            f2=inner_folder_path.split("\\")[-1]
                            # 构造最终路径
                            # return os.path.join(inner_folder_path, filename)
                            return f1+"/"+f2

    return None  # 如果没有找到合适的文件夹




# 定义文件保存路径
# TRANSUPLOAD_DIRECTORY = "C:/Users/86132/Desktop/28P-46PV3DRAW"
# # 确保 upload 目录存在
# os.makedirs(TRANSUPLOAD_DIRECTORY, exist_ok=True)
@app.post('/api/singleConvert/')  # 注意 API 路径前面需要加斜杠
async def upload_file(file: UploadFile = File(...)):
    # 定义数据库URL
    #  SQLALCHEMY_DATABASE_URLg = "mysql+pymysql://root:root@localhost/human_neuron"
    SQLALCHEMY_DATABASE_URLg = "mysql+pymysql://root:hneuronbyseu123@localhost/human_neuron"

    # 创建数据库引擎
    engineg = create_engine(SQLALCHEMY_DATABASE_URLg)

    # 创建MetaData对象
    metadatag = MetaData()

    # 反射目标表
    tracking_table = Table('human_singlecell_trackingtable_20240712', metadatag, autoload_with=engineg)

    # 创建会话
    Session = sessionmaker(bind=engineg)
    session = Session()

    # 第一步，确定上传路径
    uploadbase="/mnt/nfs/hndb/V3DRAW_16bit" # 16bit的根目录
    result_path = find_storage_path(uploadbase, file.filename)
    parts = result_path.split('//')

    # 然后从第二部分中提取所需的内容
    result_path = parts[1].split('/')[-2] + '/' + parts[1].split('/')[-1]
    print("1result path",result_path)
    tmp = os.path.join(uploadbase,result_path).replace("\\", "/")
    print(tmp)
    print(file.filename)
    neuronImage = os.path.join(tmp,file.filename).replace("\\", "/")
    print(neuronImage)
    # # 检查目录是否存在
    # if not neuronImage:  # 若文件不存在，则开始上传文件
    #     with open(neuronImage, "wb") as buffer:
    #         shutil.copyfileobj(file.file, buffer)
    with open(neuronImage, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = os.path.basename(neuronImage)

    # 获取文件名之前的路径部分
    directory = os.path.dirname(neuronImage)

    # 转MIP

    dbmip1=os.path.join("MIP_Downsample",result_path).replace("\\", "/")
    dbmip2=os.path.join(dbmip1,image.replace(".v3draw",'.tif')).replace("\\", "/")
    stmt = (
        update(tracking_table)
            .where(tracking_table.c['Cell ID'] == image.split("_")[0])  # 这里假设表中有一个主键id列
            .values(image_file=dbmip2)
    )
    session.execute(stmt)


    mipbase=r"/mnt/nfs/hndb/MIP_Downsample"
    directory1=os.path.join(mipbase,result_path).replace("\\", "/")

    outImage=os.path.join(directory1,image.replace(".v3draw",'.tif')).replace("\\", "/")
    # 创建文件
    with open(outImage, 'w') as file:
        pass  # 不执行任何操作
    pp='1:1:e'

    cmd = f'xvfb-run -a -s "-screen 0 640x480x16" "/vaa3d/Vaa3D-x.1.1.4Ubuntu/Vaa3D-x" -x mipZSlices -f mip_zslices -i {neuronImage} -o {outImage} -p 1:1:e'
    os.system(cmd)

    # 转8bit
    bit8base = r"/mnt/nfs/hndb/V3DRAW_8bit"
    directory2 = os.path.join(bit8base, result_path).replace("\\", "/")
    outImage=os.path.join(directory2,'8bit_'+image).replace("\\", "/")
    # 创建文件
    with open(outImage, 'w') as file:
        pass  # 不执行任何操作

    cmd1 = f'xvfb-run -a -s "-screen 0 640x480x16" "/vaa3d/Vaa3D-x.1.1.4Ubuntu/Vaa3D-x" -x datatypeconvert -f dtc -i {neuronImage} -o {outImage} -p 1'
    os.system(cmd1)

    # 转pbd

    dbpbd1=os.path.join("V3DPBD",result_path).replace("\\", "/")
    dbpbd2=os.path.join(dbpbd1,image.replace(".v3draw",'.v3dpbd')).replace("\\", "/")
    stmt = (
        update(tracking_table)
            .where(tracking_table.c['Cell ID'] == image.split("_")[0])  # 这里假设表中有一个主键id列
            .values(v3dpbd_file=dbpbd2)
    )
    session.execute(stmt)

    session.commit()
    # 关闭会话
    session.close()

    pbdbase = r"/mnt/nfs/hndb/V3DPBD"
    directory3 = os.path.join(pbdbase, result_path).replace("\\", "/")
    outImage=os.path.join(directory3,image.replace(".v3draw",'.v3dpbd')).replace("\\", "/")
    # 创建文件
    with open(outImage, 'w') as file:
        pass  # 不执行任何操作
    cmd2 = f'xvfb-run -a -s "-screen 0 640x480x16" "/vaa3d/Vaa3D-x.1.1.4Ubuntu/Vaa3D-x" -x convert_file_format -f convert_format -i {neuronImage} -o {outImage}'

    os.system(cmd2)


    # 重命名
    new_file_name = tmp+"/"+image.split("_")[0]+".v3draw"
    print(new_file_name)
    # 重命名文件
    os.rename(neuronImage, new_file_name)
    # return {"filename": file.filename, "message": "文件上传成功"}
'''**********************************2D图像上传，拼接用，开始******************************************'''
@app.post("/api/2Dupload/")
async def upload_2Dfiles(folderName: str = Form(...), files: List[UploadFile] = File(...)):

    # 还要
    # from fastapi import FastAPI, File, UploadFile, Form

    # UPLOAD_DIR = f"C:/Users/86132/Desktop/MIP_down/2d-batch/{folderName}"
    UPLOAD_DIR ="/mnt/nfs/hndb/2D_raw_images"
    # 第一步，确定上传路径
    result_path = find_storage_path(UPLOAD_DIR, folderName)
    parts = result_path.split('//')

    # 然后从第二部分中提取所需的内容
    result_path = parts[1].split('/')[-2] + '/' + parts[1].split('/')[-1]
    print("1result path",result_path)
    tmp = os.path.join(UPLOAD_DIR,result_path).replace("\\", "/")  # tmp是mnt下的目录
    print(tmp)
    UPLOAD_DIR = os.path.join(tmp,folderName).replace("\\", "/")



    # 创建上传目录
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_files = []
    for file in files:
        file_location = os.path.join(UPLOAD_DIR, os.path.basename(file.filename)).replace("\\", '/')
        with open(file_location, "wb") as f:
            f.write(await file.read())
        saved_files.append(file_location)

    return JSONResponse(content={"uploaded_files": saved_files})

# @app.post("/api/2Dupload")
# async def upload_files(files: List[UploadFile] = File(...)):
#     UPLOAD_DIR = "C:/Users/86132/Desktop/MIP_down/2d-batch"
#     # 创建上传目录
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     saved_files = []
#     for file in files:  # 当 file_location 路径不存在时：
#                         # 代码会创建一个新的文件，文件路径为 file_location，并将上传的文件内容写入其中。
#                         # 当 file_location 路径存在时：
#                         # 代码将覆盖已存在的文件，新的文件内容将替换原有文件的内容。这意味着之前的内容会丢失。
#         file_location = os.path.join(UPLOAD_DIR, os.path.basename(file.filename)).replace("\\",'/')
#         with open(file_location, "wb") as f:
#             f.write(await file.read())
#         saved_files.append(file_location)
#
#     return JSONResponse(content={"uploaded_files": saved_files})
'''**********************************2D图像上传，拼接用，结束******************************************'''


@app.post("/api/register/")
def register(user: schemas.RegisterModel, db: Session = Depends(get_db)):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"msg": "User created successfully"}

@app.post("/api/login/")
def login(login: schemas.LoginModel, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login.username).first()
    if not user:
        logging.error(f"User {login.username} not found")
        raise HTTPException(status_code=401, detail="Bad username or password")

    if not pwd_context.verify(login.password, user.hashed_password):
        logging.error(f"Password for user {login.username} is incorrect")
        raise HTTPException(status_code=401, detail="Bad username or password")

    # access_token = Authorize.create_access_token(subject=str(user.id))

    # 获取过期时间
    access_token_expires = timedelta(minutes=60*24)  # 设置 token 有效期为 1 天
    access_token = Authorize.create_access_token(subject=str(user.id), expires_time=access_token_expires)

    crud.create_user_log(db, user.id, "User logged in")
    return {
        "access_token": access_token,
        "expires_in": access_token_expires.total_seconds()  # 返回过期时间（秒）
    }

@app.get("/api/protected/")
def protected(Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()
    return {"msg": "You are logged in"}

# @app.get("/download/{file_path:path}")
# def download_file(file_path: str):
#     if os.path.exists(file_path):
#         return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type='application/octet-stream')
#     else:
#         raise HTTPException(status_code=404, detail="File not found")

# 下载V3DPBD
@app.get("/api/download")
def download_file(file_path: str, cell_id: str, db: Session = Depends(get_db)):
    # 提取文件名和路径
    file_path = '/mnt/nfs/hndb/' + file_path
    file_name = os.path.basename(file_path)
    directory = os.path.dirname(file_path)
    print(f"Request to download file: {file_path} with cell_id: {cell_id} \nfile_name: {file_name} \ndirectory: {directory} \n")

    # 从数据库获取数据
    cell_data = crud.get_single_cell_data_by_id(db, cell_id)
    print(f"Extracted cell_id: {cell_id}, cell_data: {cell_data}")
    print(f"soma_x, soma_y, soma_z: {cell_data.soma_x}, {cell_data.soma_y}, {cell_data.soma_z} \n")

    if not cell_data:
        raise HTTPException(status_code=404, detail="Cell data not found")

    if cell_data.image_cell_id == '-' or \
            (cell_data.image_cell_id != '-' and (cell_data.soma_x == '-' or cell_data.soma_y == '-' or cell_data.soma_z == '-')):
        # 直接下载 .v3dpbd 文件
        if os.path.exists(file_path):
            return FileResponse(path=file_path, filename=file_name, media_type='application/octet-stream')
        else:
            raise HTTPException(status_code=404, detail="File not found")
    else:
        # 检查 temp 目录中是否有对应的 {cell_id}.zip 文件
        zip_file_path = os.path.join('/mnt/nfs/hndb/temp', f"{cell_id}.zip")
        if os.path.exists(zip_file_path):
            # 如果有，直接返回 zip 文件
            print(f"{cell_id}.zip is in the temp folder")
            return FileResponse(path=zip_file_path, filename=os.path.basename(zip_file_path), media_type='application/zip')

        # 生成 .marker 文件
        marker_file_path = os.path.join('/mnt/nfs/hndb/temp', f"{cell_id}.marker")
        with open(marker_file_path, 'w') as marker_file:
            marker_file.write("##x,y,z,radius,shape,name,comment,color_r,color_g,color_b\n")
            marker_file.write(f"{cell_data.soma_x},{cell_data.soma_y},{cell_data.soma_z},0,0,{cell_data.cell_id},0,255,0,0\n")

        # 打包 .v3dpbd 和 .marker 文件
        zip_file_path = os.path.join('/mnt/nfs/hndb/temp', f"{cell_id}.zip")
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            zipf.write(file_path, file_name)
            zipf.write(marker_file_path, os.path.basename(marker_file_path))

        # 删除 .marker 文件
        os.remove(marker_file_path)
        # os.remove(zip_file_path)

        # 返回打包文件
        return FileResponse(path=zip_file_path, filename=os.path.basename(zip_file_path), media_type='application/zip')

# 获取MIP
@app.get("/api/image/{file_path:path}")
def get_image(file_path: str):
    try:
        file_path = '/mnt/nfs/hndb/'+file_path
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # 如果文件是 .tif 格式，转换为 .jpg
        if file_path.endswith('.tif'):
            jpg_file_path = file_path.replace('.tif', '.jpg')
            if not os.path.exists(jpg_file_path):
                with Image.open(file_path) as img:
                    img.convert("RGB").save(jpg_file_path, "JPEG")

            return FileResponse(jpg_file_path)
        else:
            return FileResponse(file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 搜索 + 获取数据
@app.get("/api/singlecell/", response_model=dict)
def read_single_cell_data(
        skip: int = 0,
        limit: int = 20,
        cell_id_start: Optional[str] = Query(None),
        cell_id_end: Optional[str] = Query(None),
        patient_number: Optional[str] = Query(None),
        tissue_block_number: Optional[str] = Query(None),
        slice_number: Optional[str] = Query(None),
        slicing_method: Optional[str] = Query(None),
        fresh_perfusion: Optional[str] = Query(None),
        brain_region: Optional[str] = Query(None),
        immunohistochemistry: Optional[str] = Query(None),
        inject_method: Optional[str] = Query(None),
        db: Session = Depends(get_db)
):
    query_params = {
        'cell_id_start': cell_id_start,
        'cell_id_end': cell_id_end,
        'patient_number': patient_number.split(',') if patient_number else [],
        'tissue_block_number': tissue_block_number.split(',') if tissue_block_number else [],
        'slice_number': slice_number.split(',') if slice_number else [],
        'slicing_method': slicing_method,
        'fresh_perfusion': fresh_perfusion,
        'brain_region': brain_region.split(',') if brain_region else [],
        'immunohistochemistry': immunohistochemistry,
        'inject_method': inject_method
    }
    data = crud.get_single_cell_data(db, skip=skip, limit=limit, query_params=query_params)
    total = crud.get_total_count(db, query_params=query_params)
    return {
        "data": [schemas.HumanSingleCellTrackingTable.from_orm(item) for item in data],
        "total": total
    }

@app.get("/api/get-options")
def get_options(db: Session = Depends(get_db)):
    sample_id_options = db.query(models.HumanSingleCellTrackingTable.patient_number).distinct().order_by(asc(models.HumanSingleCellTrackingTable.patient_number)).all()
    tissue_id_options = db.query(models.HumanSingleCellTrackingTable.tissue_block_number).distinct().order_by(asc(models.HumanSingleCellTrackingTable.tissue_block_number)).all()
    slice_id_options = db.query(models.HumanSingleCellTrackingTable.slice_number).distinct().order_by(asc(models.HumanSingleCellTrackingTable.slice_number)).all()
    brain_region_options = db.query(models.HumanSingleCellTrackingTable.brain_region).distinct().order_by(asc(models.HumanSingleCellTrackingTable.brain_region)).all()

    return {
        "sample_id_options": [{"value": option[0], "label": option[0]} for option in sample_id_options],
        "tissue_id_options": [{"value": option[0], "label": option[0]} for option in tissue_id_options],
        "slice_id_options": [{"value": option[0], "label": option[0]} for option in slice_id_options],
        "brain_region_options": [{"value": option[0], "label": option[0]} for option in brain_region_options]
    }

# 上传数据界面的默认值（上一个Cell ID的键值）
@app.get("/api/defaultvalues/", response_model=dict)
def read_default_values(db: Session = Depends(get_db)):
    try:
        defaults = crud.get_default_values(db)
        return {
            "cell_id": defaults["cell_id"],
            "last_record": defaults["last_record"]
        }
    except Exception as e:
        logging.error(f"Error reading default values: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 定义数据库模型字段的顺序
field_order = [
    "cell_id", "patient_number", "tissue_block_number", "small_number",
    "slice_number", "confirmed", "slicing_method", "slicing_and_fixation_order", "pre_fixation_days",
    "post_fixation_days", "sample_preparation_staff", "sample_preparation_date", "fresh_perfusion",
    "post_perfusion_4percent_pfa", "post_perfusion_10percent_formalin", "brain_region", "slice_thickness",
    "confirmed_0", "tissue_dissection_time", "perfusion_start_time", "perfusion_end_time", "after_surgery_hours",
    "cell_depth", "perfusion_current", "perfusion_time_on", "perfusion_time_off", "dye_name", "dye_concentration",
    "experiment_temperature", "experiment_humidity", "inject_method", "perfusion_date", "perfusion_staff", "ihc_category"
                                                                                                           "immunohistochemistry", "first_antibody_concentration", "secondary_antibody_band",
    "dapi_concentration", "laser_wavelength", "laser_power", "laser_power_ratio", "pmt_voltage", "z_size",
    "tiling", "overlap", "xy_resolution", "z_resolution", "document_name", "image_cell_id", "shooting_date",
    "shooting_staff", "image_size", "confirmed_1", "reconstruction_staff", "status", "inspection_staff", "status_0",
    "sealed_slide", "status_1", "dye_solvent", "remarks", "image_file", "v3dpbd_file", "soma_x", "soma_y", "soma_z"
]

def sort_dict_by_order(data, order):
    return OrderedDict((key, data[key]) for key in order if key in data)

# 上传数据
@app.post("/api/singlecell/", response_model=schemas.HumanSingleCellTrackingTable)
def create_single_cell_data(data: schemas.HumanSingleCellTrackingTableCreate, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    print("create_single_cell_data")
    try:
        try:
            # 验证JWT
            Authorize.jwt_required()
            print("JWT validation passed")
            logging.info("JWT validation passed")
        except Exception as jwt_error:
            print(f"JWT validation failed: {jwt_error}")
            logging.error(f"JWT validation failed: {jwt_error}")
            raise HTTPException(status_code=401, detail="JWT validation failed")

        user_id = Authorize.get_jwt_subject()
        print(f"JWT subject: {user_id}")
        logging.info(f"JWT subject: {user_id}")
        logging.info(f"User {user_id} is creating single cell data")
        single_cell_data = crud.create_single_cell_data(db=db, data=data)
        sorted_details = sort_dict_by_order(data.dict(), field_order)
        details = details=json.dumps(sorted_details)
        crud.create_user_log(db, int(user_id), f"Create single cell data with id {single_cell_data.cell_id}", details=details)

        return single_cell_data
    except Exception as e:
        logging.error(f"Error creating single cell data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 删除数据
@app.delete("/api/singlecell/{cell_id}", response_model=schemas.HumanSingleCellTrackingTable)
def delete_single_cell_data(cell_id: str, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    try:
        Authorize.jwt_required()
        user_id = Authorize.get_jwt_subject()  # 确保 user_id 是整数类型
        logging.info(f"User {user_id} is deleting single cell data with id {cell_id}")

        # 查询删除前的数据
        single_cell_data = crud.get_single_cell_data_by_id(db, cell_id)
        if single_cell_data is None:
            raise HTTPException(status_code=404, detail="数据未找到")

        # 记录删除前的数据内容
        deleted_data_details = single_cell_data.__dict__.copy()
        del deleted_data_details['_sa_instance_state']
        sorted_details = sort_dict_by_order(deleted_data_details, field_order)
        details=json.dumps(sorted_details)

        # 执行删除操作
        deleted_data = crud.delete_single_cell_data(db=db, cell_id=cell_id)

        # 创建删除日志，记录详细信息
        crud.create_user_log(db, int(user_id), f"Delete single cell data with id {cell_id}", details=details)
        return deleted_data
    except Exception as e:
        logging.error(f"Error deleting single cell data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 编辑数据
@app.put("/api/singlecell/{cell_id}", response_model=schemas.HumanSingleCellTrackingTable)
def update_single_cell_data(cell_id: str, data: schemas.HumanSingleCellTrackingTableCreate, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    try:
        Authorize.jwt_required()
        user_id = Authorize.get_jwt_subject()    # 确保 user_id 是整数类型
        logging.info(f"User {user_id} is updating single cell data with id {cell_id}")
        db_data = crud.get_single_cell_data_by_id(db, cell_id)

        if db_data is None:
            raise HTTPException(status_code=404, detail="数据未找到")

        # 记录原始数据
        original_data = db_data.__dict__.copy()
        del original_data['_sa_instance_state']

        updated_data = crud.update_single_cell_data(db=db, db_obj=db_data, update_data=data)

        # 记录修改后的数据
        changes = {
            "original": sort_dict_by_order(original_data, field_order),
            "updated": sort_dict_by_order(data.dict(), field_order)
        }

        details=json.dumps(changes)
        crud.create_user_log(db, int(user_id), f"Update single cell data with id {cell_id}", details=details)
        return updated_data
    except Exception as e:
        logging.error(f"Error updating single cell data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 数据库状态
@app.get("/api/datastatus/")
def get_data_status(db: Session = Depends(get_db)):
    try:
        total_samples = crud.get_total_samples(db)
        valid_samples = crud.get_valid_samples(db)
        total_cells = crud.get_total_count(db)
        total_regions = crud.get_total_regions(db)
        return {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "cells": total_cells,
            "regions": total_regions
        }
    except Exception as e:
        logging.error(f"Error getting data status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/api/productiontrend/", response_model=dict)
def get_production_trend(start_date: str = None, end_date: str = None, db: Session = Depends(get_db)):
    try:
        query = db.query(models.HumanSingleCellTrackingTable)
        if start_date and end_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            query = query.filter(
                models.HumanSingleCellTrackingTable.perfusion_date >= start_date_obj,
                models.HumanSingleCellTrackingTable.perfusion_date <= end_date_obj
            )

        result = query.all()

        daily_counts = {}
        total_counts = {}
        for record in result:
            date_str = record.perfusion_date
            date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
            if date in daily_counts:
                daily_counts[date] += 1
            else:
                daily_counts[date] = 1

            if date in total_counts:
                total_counts[date] += daily_counts[date]
            else:
                total_counts[date] = daily_counts[date]

        dates = sorted(daily_counts.keys())
        daily_values = [daily_counts[date] for date in dates]
        total_values = [sum(daily_values[:i+1]) for i in range(len(daily_values))]

        min_date_str = db.query(func.min(models.HumanSingleCellTrackingTable.perfusion_date)).scalar()
        max_date_str = db.query(func.max(models.HumanSingleCellTrackingTable.perfusion_date)).scalar()

        min_date = datetime.strptime(min_date_str, '%Y-%m-%d') if min_date_str else None
        max_date = datetime.strptime(max_date_str, '%Y-%m-%d') if max_date_str else None

        return {
            "dates": dates,
            "totalValues": total_values,
            "dailyValues": daily_values,
            "minDate": min_date.strftime('%Y-%m-%d') if min_date else None,
            "maxDate": max_date.strftime('%Y-%m-%d') if max_date else None
        }
    except Exception as e:
        logging.error(f"Error fetching production trend: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/api/age-distribution")
def get_age_distribution(db: Session = Depends(get_db)):
    age_groups = {
        "小于40岁": 0,
        "40-60岁": 0,
        "大于60岁": 0
    }

    # Query the database for age and patient number
    data = db.query(models.Sample_Information.patient_age, models.Sample_Information.patient_number) \
        .filter(models.Sample_Information.patient_number != '--',
                models.Sample_Information.patient_number != '-',
                models.Sample_Information.patient_number != '',
                models.Sample_Information.patient_age != '待补充',
                models.Sample_Information.patient_age != '',
                models.Sample_Information.patient_age != '-',
                models.Sample_Information.patient_age != '--'
                ).all()

    # Use a set to keep track of unique patient numbers
    unique_patients = set()

    for age, patient_number in data:
        if patient_number not in unique_patients:
            unique_patients.add(patient_number)
            if age.isdigit():
                age_int = int(age)
                if age_int < 40:
                    age_groups["小于40岁"] += 1
                elif 40 <= age_int <= 60:
                    age_groups["40-60岁"] += 1
                else:
                    age_groups["大于60岁"] += 1

    return [{"name": k, "value": v} for k, v in age_groups.items()]

@app.get("/api/brain-region-distribution")
def get_brain_region_distribution(db: Session = Depends(get_db)):
    brain_region_data = db.query(
        models.HumanSingleCellTrackingTable.brain_region,
        func.count(models.HumanSingleCellTrackingTable.id).label('count')
    ).group_by(models.HumanSingleCellTrackingTable.brain_region).order_by(func.count(models.HumanSingleCellTrackingTable.id).desc()).limit(10).all()
    return {
        "categories": [item[0] for item in brain_region_data],
        "data": [item[1] for item in brain_region_data]
    }

@app.get("/api/immunohistochemistry-distribution")
def get_immunohistochemistry_distribution(db: Session = Depends(get_db)):
    ihc_data = db.query(
        models.HumanSingleCellTrackingTable.immunohistochemistry,
        func.count(models.HumanSingleCellTrackingTable.id).label('count')
    ).group_by(models.HumanSingleCellTrackingTable.immunohistochemistry).all()

    result = []
    for item in ihc_data:
        if item[0] == "1":
            result.append({"name": "ihc", "value": item[1]})
        elif item[0] == "0":
            result.append({"name": "None_ihc", "value": item[1]})
        else:
            result.append({"name": "未知", "value": item[1]})

    return result

@app.get("/api/sample-source-distribution")
def get_sample_source_distribution(db: Session = Depends(get_db)):
    # Query to get unique sample_id and patient_number where patient_number is not '--'
    data = db.query(models.Sample_Information.sample_id, models.Sample_Information.patient_number) \
        .filter(models.Sample_Information.patient_number != '--',
                models.Sample_Information.patient_number != '-',
                models.Sample_Information.patient_number != ''
                ).distinct().all()

    source_count = {}
    source_patient_numbers = {}  # Dictionary to track patient_numbers for each source
    source_cell_count = {}  # Dictionary to track HumanSingleCellTrackingTable counts for each source
    source_brain_region_distribution = {}  # Dictionary to track brain region distribution for each source
    unique_patient_numbers = set()  # Set to track unique patient numbers

    for item in data:
        sample_id, patient_number = item
        if patient_number not in unique_patient_numbers:
            unique_patient_numbers.add(patient_number)
            source = '-'.join(sample_id.split('-')[:2])  # Extract content before the second dash
            if source not in source_count:
                source_count[source] = 0
                source_patient_numbers[source] = []  # Initialize a list for each new source
                source_cell_count[source] = 0  # Initialize the count for tracking table data
                source_brain_region_distribution[source] = {}  # Initialize the brain region distribution
            source_count[source] += 1
            source_patient_numbers[source].append(patient_number)  # Add the patient_number to the corresponding source

    # Query HumanSingleCellTrackingTable and count the entries for each patient_number
    patient_cell_count = {}
    for source, patient_numbers in source_patient_numbers.items():
        for patient_number in patient_numbers:
            # Query to get brain region distribution
            tracking_data = db.query(models.HumanSingleCellTrackingTable.brain_region) \
                .filter(models.HumanSingleCellTrackingTable.patient_number == patient_number) \
                .all()

            tracking_count = len(tracking_data)
            patient_cell_count[patient_number] = tracking_count
            source_cell_count[source] += tracking_count  # Add to the corresponding source's total tracking count

            # Update brain region distribution
            for brain_region in tracking_data:
                region = brain_region[0]  # Extract the brain_region from the query result
                if region not in source_brain_region_distribution[source]:
                    source_brain_region_distribution[source][region] = 0
                source_brain_region_distribution[source][region] += 1

    return {
        "categories": list(source_count.keys()),
        "data": list(source_count.values()),
        "source_patient_numbers": source_patient_numbers,  # Include the patient number mapping
        "patient_cell_count": patient_cell_count,  # Count of tracking table entries per patient number
        "source_cell_count": source_cell_count,  # Count of tracking table entries per source
        "source_brain_region_distribution": source_brain_region_distribution  # Brain region distribution per source
    }

# @app.get("/api/cell-source-distribution")
# def get_cell__source_distribution(db: Session = Depends(get_db)):


@app.post("/api/savereport/")
def save_report(report: schemas.Report, db: Session = Depends(get_db)):
    db_report = DailyReport(report_date=report.report_date, content=report.content)
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return {"msg": "Report saved successfully"}

@app.get("/api/latestreport/")
def get_latest_report(db: Session = Depends(get_db)):
    latest_report = db.query(DailyReport).order_by(desc(DailyReport.report_date)).first()
    if not latest_report:
        raise HTTPException(status_code=404, detail="No report found")
    return {
        "report_date": latest_report.report_date,
        "content": latest_report.content
    }

@app.get("/api/sample_information/", response_model=dict)
def read_sample_information(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    data = db.query(models.Sample_Information).offset(skip).limit(limit).all()
    total = db.query(models.Sample_Information).count()
    return {
        "data": [schemas.SampleInfo.from_orm(item) for item in data],
        "total": total
    }

# 上传数据界面的默认值（上一个Cell ID的键值）
@app.get("/api/defaultInfo/", response_model=dict)
def read_default_info(db: Session = Depends(get_db)):
    try:
        defaults = crud.get_default_info(db)
        return {
            "last_record": defaults["last_record"]
        }
    except Exception as e:
        logging.error(f"Error reading default INFO: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

sample_field_order = [
    "idx","total_id", "patient_id", "sample_id", "id", "patient_number", "tissue_id",
    "sample_slice_info", "perfusion_info", "perfusion_evaluation", "patient_age", "gender",
    "surgery_date", "sample_size", "tumor_location", "intracranial_location", "english_abbr_nj",
    "english_full_name", "left_right_brain", "sample_processing_method", "removal_time",
    "immersion_time", "storage_location", "sampling_method_personnel", "send_sample_date",
    "treatment_naive", "treatment_received", "pathological_diagnosis"
]

# 上传
@app.post("/api/sample_information", response_model=schemas.SampleInfo)
def create_sample(sample: schemas.SampleInfoCreate, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    try:
        Authorize.jwt_required()
        user_id = Authorize.get_jwt_subject()

        if not sample.total_id:
            raise HTTPException(status_code=400, detail="Total ID cannot be null")

        db_sample = models.Sample_Information(**sample.dict())
        db.add(db_sample)
        db.commit()
        db.refresh(db_sample)

        sorted_details = sort_dict_by_order(sample.dict(), sample_field_order)
        details = json.dumps(sorted_details, ensure_ascii=False)   # ensure_ascii=False 用来正确处理中文内容，确保中文字符不被转义
        crud.create_user_log(db, int(user_id), f"Create sample information with idx {db_sample.idx}", details=details)

        return db_sample
    except Exception as e:
        logging.error(f"Error creating sample information: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 编辑
@app.put("/api/sample_information/{idx}", response_model=schemas.SampleInfo)
def update_sample_information(idx: int, updated_info: schemas.SampleInfoCreate, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    try:
        Authorize.jwt_required()
        user_id = Authorize.get_jwt_subject()

        sample_info = db.query(models.Sample_Information).filter(models.Sample_Information.idx == idx).first()
        if not sample_info:
            raise HTTPException(status_code=404, detail="Sample not found")

        original_data = sample_info.__dict__.copy()
        del original_data['_sa_instance_state']

        for var, value in vars(updated_info).items():
            setattr(sample_info, var, value) if value is not None else None
        db.commit()
        db.refresh(sample_info)

        changes = {
            "original": sort_dict_by_order(original_data, sample_field_order),
            "updated": sort_dict_by_order(updated_info.dict(), sample_field_order)
        }
        details = json.dumps(changes, ensure_ascii=False)
        crud.create_user_log(db, int(user_id), f"Update sample information with idx {idx}", details=details)

        return sample_info
    except Exception as e:
        logging.error(f"Error updating sample information: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 删除
@app.delete("/api/sample_information/{idx}", response_model=schemas.SampleInfo)
def delete_sample_information(idx: int, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    try:
        Authorize.jwt_required()
        user_id = Authorize.get_jwt_subject()

        sample_info = db.query(models.Sample_Information).filter(models.Sample_Information.idx == idx).first()
        if not sample_info:
            raise HTTPException(status_code=404, detail="Sample not found")

        deleted_data_details = sample_info.__dict__.copy()
        del deleted_data_details['_sa_instance_state']
        sorted_details = sort_dict_by_order(deleted_data_details, sample_field_order)
        details = json.dumps(sorted_details, ensure_ascii=False)

        db.delete(sample_info)
        db.commit()

        crud.create_user_log(db, int(user_id), f"Delete sample information with idx {idx}", details=details)
        return sample_info
    except Exception as e:
        logging.error(f"Error deleting sample information: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 文件上传路径
UPLOAD_DIR = "/mnt/nfs/hndb/Record_Book_Pics"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 静态文件路径
app.mount("/Record_Book_Pics", StaticFiles(directory=UPLOAD_DIR), name="Record_Book_Pics")


@app.post("/api/upload_pics")
async def upload_image(file: UploadFile = File(...)):
    # 文件类型检查
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    # 如果文件已经存在，先删除旧文件
    # if os.path.exists(file_path):
    #    os.remove(file_path)

    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    return JSONResponse(content={"filename": file.filename})

@app.get("/api/record_book_pics")
async def get_record_book_pics():
    try:
        # 获取所有图片文件名
        pics = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 按文件名从小到大排序
        pics.sort()  # 或者使用 pics = sorted(pics)

        return JSONResponse(content={"pics": pics})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch record book pics")

#@app.get("/api/record_book_pics")
#async def get_record_book_pics():
# try:
#     pics = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
#      return JSONResponse(content={"pics": pics})
#   except Exception as e:
#        raise HTTPException(status_code=500, detail="Failed to fetch record book pics")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


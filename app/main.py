# FastAPI应用的入口，包含路由定义和数据库连接配置
import csv
import mimetypes
import shutil
from io import StringIO

import cv2
import imageio
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form, Header, Request, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import func, desc, asc, Integer, create_engine, MetaData, Table, update, select, insert, cast,text
from fastapi.middleware.cors import CORSMiddleware
from . import models, schemas, crud, foundPBD
from .database import SessionLocal
from fastapi.staticfiles import StaticFiles
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from PIL import Image
import os
from app.models import User, DailyReport, SamplePreparation, ImagingRecord
from .schemas import SamplePreparationSchema, ImagingRecordSchema
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
import uvicorn
import json
from collections import OrderedDict, Counter
import zipfile
import aiofiles
import shutil
import sys
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import re
from sqlalchemy.exc import SQLAlchemyError
import redis
import openai
import parsedatetime
import spacy

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 static 目录，服务静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


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


'''**********************************样本图像数据统计*****************************************'''


@app.post("/api/generate_sample_xlsx")
async def generate_xlsx(request: Request, db: Session = Depends(get_db)):
    request_data = await request.json()
    hospital_name = request_data.get("hospital")

    # 如果 hospital_name 是逗号分隔字符串，转为列表
    if hospital_name and isinstance(hospital_name, str):
        hospital_name = hospital_name.split(",")

    # 从数据库中获取所有样本
    hossamples = db.query(models.Sample_Information).all()

    # 根据 hospital_name 筛选
    if not hospital_name or hospital_name == "none":
        samples = hossamples
    else:
        samples = [
            sample
            for sample in hossamples
            if "-".join(sample.sample_id.split("-")[:2]) in hospital_name
        ]

    # 准备 XLSX 数据
    xlsx_data = {
        "hospital": [],
        "sample_id": [],
        "loss": [],
        "sample_snapshot": [],
        "sample_image": [],
        "sample_annotation": [],
    }

    for sample in samples:
        sample_id = sample.sample_id
        hospital_value = "-".join(sample_id.split("-")[:2])

        xlsx_data["hospital"].append(hospital_value)
        xlsx_data["sample_id"].append(sample_id)
        xlsx_data["sample_snapshot"].append(0 if not sample.sample_snapshot else 1)
        xlsx_data["sample_image"].append(0 if not sample.sample_image else 1)
        xlsx_data["sample_annotation"].append(0 if not sample.sample_annotation else 1)
        xlsx_data["loss"].append(
            1
            if all(
                [
                    sample.sample_snapshot,
                    sample.sample_image,
                    sample.sample_annotation,
                ]
            )
            else 0
        )

    df = pd.DataFrame(xlsx_data)
    xlsx_file_path = "sample_data.xlsx"
    df.to_excel(xlsx_file_path, index=False)

    return StreamingResponse(
        open(xlsx_file_path, mode="rb"),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=sample_data.xlsx"},
    )


'''**********************************样本信息上传*********************************************
新增三列
    sample_snapshot = Column("sample_snapshot", String(255))
    sample_image = Column("sample_image", String(255))
    sample_annotation = Column("sample_annotation", String(255))

**************************'''


@app.post("/api/Upload_Sample_snapshot")
async def Upload_Sample_snapshot(folderName: str = Form(...), sample_idx: int = Form(...),
                                 files: list[UploadFile] = File(...), Authorize: AuthJWT = Depends(),
                                 db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()  # 确保 user_id 是整数类型

    # 按照'-'分割字符串
    parts = folderName.split('-')
    # 判断第一个元素是否是数字
    if parts[0].isdigit():
        # 如果是数字，则只保留剩下的部分
        realfolderName = '-'.join(parts[1:])
    else:
        # 否则，直接打印原字符串
        realfolderName = folderName

    details = f"{str(sample_idx)}_{realfolderName}_sample_snapshot"
    crud.create_user_log(db, int(user_id), action=f"upload sample snapshot",
                         details=details)

    base_upload_dir = f"/mnt/nfs/hndb/Sample_Files/{str(sample_idx)}_{realfolderName}/sample_snapshot"
    os.makedirs(base_upload_dir, exist_ok=True)
    responses = []
    for file in files:
        # 使用 webkitRelativePath 获取相对路径，假设文件名为 "folderName/innerFolder/file.txt"
        relative_path = file.filename.split('/', 1)[-1]  # 跳过最外层文件夹

        file_location = os.path.join(base_upload_dir, relative_path)
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        responses.append(file_location)

        # 在数据库中更新条目
    db.query(models.Sample_Information) \
        .filter(models.Sample_Information.idx == sample_idx) \
        .update({models.Sample_Information.sample_snapshot: realfolderName})
    # 提交更改到数据库
    db.commit()
    return JSONResponse(content={"message": "Upload successful!", "files": responses})


@app.post("/api/Upload_Sample_image")
async def Upload_Sample_image(folderName: str = Form(...), sample_idx: int = Form(...),
                              files: list[UploadFile] = File(...), Authorize: AuthJWT = Depends(),
                              db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()  # 确保 user_id 是整数类型

    # 按照'-'分割字符串
    parts = folderName.split('-')
    # 判断第一个元素是否是数字
    if parts[0].isdigit():
        # 如果是数字，则只保留剩下的部分
        realfolderName = '-'.join(parts[1:])
    else:
        # 否则，直接打印原字符串
        realfolderName = folderName

    base_upload_dir = f"/mnt/nfs/hndb/Sample_Files/{str(sample_idx)}_{realfolderName}/sample_image"
    os.makedirs(base_upload_dir, exist_ok=True)
    responses = []
    for file in files:
        # 使用 webkitRelativePath 获取相对路径，假设文件名为 "folderName/innerFolder/file.txt"
        relative_path = file.filename.split('/', 1)[-1]  # 跳过最外层文件夹

        file_location = os.path.join(base_upload_dir, relative_path)
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        responses.append(file_location)

        # 在数据库中更新条目
    db.query(models.Sample_Information) \
        .filter(models.Sample_Information.idx == sample_idx) \
        .update({models.Sample_Information.sample_image: realfolderName})
    # 提交更改到数据库
    db.commit()

    details = f"{str(sample_idx)}_{realfolderName}_sample_image"
    crud.create_user_log(db, int(user_id), action=f"upload sample image",
                         details=details)

    return JSONResponse(content={"message": "Upload successful!", "files": responses})


@app.post("/api/Upload_Sample_annoation")
async def Upload_Sample_annoation(folderName: str = Form(...), sample_idx: int = Form(...),
                                  files: list[UploadFile] = File(...), Authorize: AuthJWT = Depends(),
                                  db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()  # 确保 user_id 是整数类型

    # 按照'-'分割字符串
    parts = folderName.split('-')
    # 判断第一个元素是否是数字
    if parts[0].isdigit():
        # 如果是数字，则只保留剩下的部分
        realfolderName = '-'.join(parts[1:])
    else:
        # 否则，直接打印原字符串
        realfolderName = folderName

    base_upload_dir = f"/mnt/nfs/hndb/Sample_Files/{str(sample_idx)}_{realfolderName}/sample_annoation"
    os.makedirs(base_upload_dir, exist_ok=True)
    responses = []
    for file in files:
        # 使用 webkitRelativePath 获取相对路径，假设文件名为 "folderName/innerFolder/file.txt"
        relative_path = file.filename.split('/', 1)[-1]  # 跳过最外层文件夹

        file_location = os.path.join(base_upload_dir, relative_path)
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        responses.append(file_location)

        # 在数据库中更新条目
    db.query(models.Sample_Information) \
        .filter(models.Sample_Information.idx == sample_idx) \
        .update({models.Sample_Information.sample_annotation: realfolderName})
    # 提交更改到数据库
    db.commit()

    details = f"{str(sample_idx)}_{realfolderName}_sample_annotation"
    crud.create_user_log(db, int(user_id), action=f"upload sample annotation",
                         details=details)
    return JSONResponse(content={"message": "Upload successful!", "files": responses})


'''**********************************样本信息下载*********************************************'''


@app.post("/api/sample_download/")
async def download_sample_file(request: Request):
    # 从请求体中获取 JSON 数据
    data = await request.json()
    idx = data.get("idx")  # 获取 idx 参数
    print(idx)

    # 确定要搜索的根文件夹路径
    root_dir = r"/mnt/nfs/hndb/Sample_Files"
    temp_dir = r"/mnt/nfs/hndb/temp/Sample_temp"

    # 检查匹配的文件夹
    matched_folder_paths = []  # 使用列表存储匹配的文件夹路径
    for folder_name in os.listdir(root_dir):
        if folder_name.split('_')[0] == str(idx):
            matched_folder_paths.append(os.path.join(root_dir, folder_name))  # 添加匹配的文件夹路径

    # 检查是否找到匹配的文件夹
    if not matched_folder_paths:
        raise HTTPException(status_code=404, detail="Folder not found")

        # 使用临时文件夹保存 ZIP 文件
    zip_file_path = os.path.join(temp_dir, f"sample_files_{idx}.zip")

    # 创建 ZIP 文件并压缩所有匹配的文件夹内容
    print("创建 ZIP 文件")
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_path in matched_folder_paths:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    # 创建相对路径以保持目录结构
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=folder_path)
                    zipf.write(file_path, arcname)

                    # 创建并返回响应
    response = FileResponse(path=zip_file_path, filename=os.path.basename(zip_file_path), media_type='application/zip')
    return response


'''**********************************一类样本图像查看*********************************************'''


@app.post("/api/view_sample_snapshots/")
async def get_sample_snapshot(request: Request):
    # 从请求体中获取 JSON 数据
    data = await request.json()
    idx = data.get("idx")  # 获取 idx 参数
    print(idx)

    FOLDER_PATH = r"/mnt/nfs/hndb/Sample_Files/"
    # 用于存储匹配的文件夹路径
    matched_folder_paths = []

    # 检查匹配的文件夹
    for folder_name in os.listdir(FOLDER_PATH):
        if folder_name.split('_')[0] == str(idx):
            matched_folder_paths.append(os.path.join(FOLDER_PATH, folder_name))  # 使用 os.path.join

    image_files = []

    # 遍历匹配的文件夹路径
    for matched_folder_path in matched_folder_paths:
        sample_snapshot_path = matched_folder_path + "/sample_snapshot"

        # 检查 sample_snapshot 目录是否存在
        if os.path.exists(sample_snapshot_path):
            # 遍历该目录中的文件
            for root, dirs, files in os.walk(sample_snapshot_path):
                for file in files:
                    if file.lower().startswith('._'):
                        continue  # 跳过该文件

                    if file.lower().endswith(('.jpg', '.jpeg','.png')):
                        # 构建完整的图像 URL
                        image_url = os.path.join(root, file).replace("\\", '/')
                        print(image_url)
                        # 使用split方法根据 '/' 分割字符串
                        # parts = image_url.split('/')
                        # 找到第一个不是空字符串的索引
                        # modified_url = '/'.join(parts[5:])  # 从索引4开始保留后面的部分
                        # 添加文件名和 URL 的字典
                        image_files.append({"name": file, "url": image_url})

            # 找到一个有效的 sample_snapshot 后就中断遍历
            break

    image_files.sort(key=lambda x: x['name'])  # 按文件名排序
    # 创建 JSON 响应
    response = JSONResponse(content={"pics": image_files})
    return response


@app.post("/api/get_sample_snapshot_url")
async def get_sample_snapshot_url(request: Request):
    # 从请求体中获取 JSON 数据
    data = await request.json()
    file_path = data.get("imagefile")  # 获取图像文件路径
    print(file_path)

    # 检查文件路径是否有效
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # 获取文件的 MIME 类型
        mime_type, _ = mimetypes.guess_type(file_path)
        return FileResponse(file_path, media_type=mime_type)  # 返回文件并设置 MIME 类型
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''****************************SWC可视化开始***************************'''


def readSWC(swc_path, mode='simple'):  # pandas DataFrame
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


def get_degree(tswc):  # Degree of node: the number of nodes connected to it
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
    rnode = swc[((swc['parent'] < 0) & (swc['type'] <= 1))]
    if rnode.shape[0] < 1:
        return -1
    return rnode.index[0]


def get_keypoint(swc, rid=None):  # keypoint: degree ≠ 2 (branches & tips)
    if rid is None:
        rid = get_rid(swc)
    # print(swc.shape)
    swc = get_degree(swc)
    idlist = swc[((swc.degree != 2) | (swc.index == rid))].index.tolist()
    return idlist


def swc2branches(swc):
    '''
    reture branch list of a swc
    '''
    keyids = get_keypoint(swc)
    branches = []
    for key in keyids:
        if (swc.loc[key, 'parent'] < 0) | (swc.loc[key, 'type'] <= 1):
            continue
        branch = []
        branch.append(key)
        pkey = swc.loc[key, 'parent']
        while True:
            branch.append(pkey)
            if pkey in keyids:
                break
            key = pkey
            if (swc.loc[key, 'parent'] < 0):
                break
            pkey = swc.loc[key, 'parent']
        branches.append(branch)
    return branches


def get_swc(swc_file, image_path, cellid, db, projection_direction='xy', ignore_background=False):
    rawID = '-'
    soma_x = 0
    soma_y = 0
    soma_z = 0

    # Query the database using the ORM session
    query = (
        db.query(models.HumanSingleCellTrackingTable)  # Replace with your actual model
        .filter(models.HumanSingleCellTrackingTable.cell_id == cellid)  # Adjust as necessary
        .with_entities(
            models.HumanSingleCellTrackingTable.image_cell_id,
            models.HumanSingleCellTrackingTable.soma_x,
            models.HumanSingleCellTrackingTable.soma_y,
            models.HumanSingleCellTrackingTable.soma_z
        )
    )

    # Execute the query
    result = query.first()  # Use first() to get a single result

    if result:
        rawID = result.image_cell_id
        soma_x = int(result.soma_x.split('.')[0])  # Convert to int after split
        soma_y = int(result.soma_y.split('.')[0])  # Convert to int after split
        soma_z = int(result.soma_z.split('.')[0])  # Convert to int after split

    background = imageio.v2.imread(image_path)
    y_size, x_size = background.shape[:2]  # 取前两维，忽略通道数

    if rawID == "-":  # 单细胞下
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

        print(y_size, x_size)
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if (ignore_background):
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

        res = swc_file.replace(".swc", ".jpg")
        finalImage = Image.fromarray(background, 'RGB')
        finalImage = finalImage.rotate(-180)
        # 进行水平镜像
        mirroredImage = finalImage.transpose(method=Image.FLIP_LEFT_RIGHT)
        mirroredImage.save(res)
        # Close the figure to free up memory

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

        print(y_size, x_size)
        if (soma_x != '-' and soma_y != '-'):
            x_start = max(int(soma_x) - 320, 0)  # 886 - 256 = 630
            y_start = max(int(soma_y) - 320, 0)  # 800 - 256 = 544
        else:
            x_start = 0
            y_start = 0
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if (ignore_background):
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
            cv2.circle(background, (int(point_l.p[1].x - x_start), int(point_l.p[1].y - y_start)), 3, color, -1)
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
                cv2.line(background, (int(x - x_start), int(y - y_start)), (int(px - x_start), int(py - y_start)),
                         colors[3], thickness)  #colors-si  -- colors[2]
            elif (projection_axes == 1):
                cv2.line(background, (x, z), (px, pz), colors[3], thickness)
            elif (projection_axes == 2):
                cv2.line(background, (y, z), (py, pz), colors[3], thickness)
        if x_size != 512 or y_size != 512:
            background = cv2.resize(background, (x_size, y_size))

        res = swc_file.replace(".swc", ".jpg")
        finalImage = Image.fromarray(background, 'RGB')
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
def get_swcimage(request: SWCfilepath, db: Session = Depends(get_db)):
    globalpath1 = "/mnt/nfs/hndb"
    repath = globalpath1 + request.ss
    mippath = globalpath1 + "/" + request.mipforswc
    swcimage = get_swc(repath, mippath, request.cellid, db=db)
    # swcimagename=os.path.basename(swcimage).split("_")[0]
    try:
        return FileResponse(swcimage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        self.s = []  # sons
        self.fn = -1  # fiber number
        self.conn = []  # connect points in other fiber
        self.mp = []  # match point in other swc
        self.neighbor = []  # neighbor closer than a distance. store neighbor number and connect info. as [d, bool]
        # self.isend = False
        self.ishead = False
        self.istail = False
        self.swcNeig = []  # neighbor closer than a distance.
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
    with open(swc_name, 'r') as f:
        lines = f.readlines()

    swcPoint_number = -1
    # swcPoint_list = []
    point_list = []
    list_map = np.zeros(500000)

    # 计数器，用于记录 point[6] == -1 的数量
    count_negative_one = 0

    for line in lines:
        if (line[0] == '#'):
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
        point[0] = swcPoint_number  # int(point[0])
        point[1] = int(point[1])
        point[2] = float(point[2])
        point[3] = float(point[3])
        point[4] = float(point[4])
        point[5] = float(point[5])
        point[6] = int(point[6])
        if (point[6] == -1):
            pass
        else:
            point[6] = int(list_map[int(point[6])]) + 1

    # swcPoint_list.append(swcPoint(0,0,0,0,0,0,0)) # an empty point numbered 0
    point_l.p.append(swcPoint(0, 0, 0, 0, 0, 0, 0))

    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        point_l.p.append(temp_swcPoint)
    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        if not temp_swcPoint.p == -1:
            # parent = swcPoint_list[int(temp_swcPoint.p)]
            parent = point_l.p[int(temp_swcPoint.p)]
            parent.s.append(temp_swcPoint.n)
        if (point[0] == 1):
            point_l.p[int(point[0])].depth = 0
        else:
            point_l.p[int(point[0])].depth = parent.depth + 1
        # point_l.p.append(temp_swcPoint)
    # for i in range(1, 10):
    #     print(point_l.p[i].s)

    return point_l  # (swcPoint_list)


def get_mip_swc(swc_file, image, cellid, db, projection_direction='xy', ignore_background=False):
    rawID = '-'
    soma_x = 0
    soma_y = 0
    soma_z = 0

    # Query the database using the ORM session
    query = (
        db.query(models.HumanSingleCellTrackingTable)  # Replace with your actual model
        .filter(models.HumanSingleCellTrackingTable.cell_id == cellid)  # Adjust as necessary
        .with_entities(
            models.HumanSingleCellTrackingTable.image_cell_id,
            models.HumanSingleCellTrackingTable.soma_x,
            models.HumanSingleCellTrackingTable.soma_y,
            models.HumanSingleCellTrackingTable.soma_z
        )
    )

    # Execute the query
    result = query.first()  # Use first() to get a single result

    if result:
        rawID = result.image_cell_id
        soma_x = int(result.soma_x.split('.')[0])  # Convert to int after split
        soma_y = int(result.soma_y.split('.')[0])  # Convert to int after split
        soma_z = int(result.soma_z.split('.')[0])  # Convert to int after split
    if rawID == '-':
        if projection_direction == 'xy':
            projection_axes = 0
        elif projection_direction == 'xz':
            projection_axes = 1
        elif projection_direction == 'yz':
            projection_axes = 2
        else:
            raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")
        image_path = image  # 注意使用原始字符串，避免转义字符问题
        background = imageio.v2.imread(image_path)
        y_size, x_size = background.shape[:2]  # 取前两维，忽略通道数
        print(y_size, x_size)
        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if (ignore_background):
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
        image_path = image  # 注意使用原始字符串，避免转义字符问题
        background = imageio.v2.imread(image_path)
        y_size, x_size = background.shape[:2]  # 取前两维，忽略通道数
        print(y_size, x_size)

        if (soma_x != '-' and soma_y != '-'):
            x_start = max(int(soma_x) - 320, 0)  # 886 - 256 = 630
            y_start = max(int(soma_y) - 320, 0)  # 800 - 256 = 544
        else:
            x_start = 0
            y_start = 0

        if x_size < 512 and y_size < 512:
            background = cv2.resize(background, (512, 512))
        #background = cv2.flip(background, 0)
        # print(image_path)
        # Resize the original image and save it back to the same path
        # original_image = imageio.v2.imread(image_path)  # 读取原始图像
        # resized_image = cv2.resize(original_image, (512, 512))
        # imageio.v2.imwrite(image_path, resized_image)  # 保存调整后的图像回原路径

        if (ignore_background):
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
            cv2.circle(background, (int(point_l.p[1].x - x_start), int(point_l.p[1].y - y_start)), 3, color, -1)
        elif (projection_axes == 1):
            cv2.circle(background, (int(point_l.p[1].x - x_start), int(point_l.p[1].z)), 3, color, -1)
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
                cv2.line(background, (int(x - x_start), int(y - y_start)), (int(px - x_start), int(py - y_start)),
                         colors[3], thickness)  #colors-si  -- colors[2]
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
def get_mipswc_image(request: MIP_SWCfilepath, db: Session = Depends(get_db)):
    globalpath = "/mnt/nfs/hndb"
    swc = globalpath + request.swc_file
    mip = globalpath + "/" + request.image_file
    # print(swc)
    #print(mip)
    re = get_mip_swc(swc, mip, request.cellid, db=db)
    finalImage = Image.fromarray(re, 'RGB')
    # 顺时针旋转 180 度
    finalImage = finalImage.rotate(-180)
    # 进行水平镜像
    mirroredImage = finalImage.transpose(method=Image.FLIP_LEFT_RIGHT)
    savepath = os.path.dirname(swc) + "/" + os.path.basename(swc)[:5] + "_Combine" + ".jpg"
    mirroredImage.save(savepath)

    try:
        return FileResponse(savepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''*******************************************MIP和SWC重叠结束*******************************'''

'''**************************************单个文件数据转换**************************************'''


def find_storage_path(base_folder, filename):
    file_number = int(filename[:5])  # 例如 "00028.txt" -> 28

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
                    f1 = outer_folder_path.split("\\")[-1]
                    inner_folder_path = os.path.join(outer_folder_path, inner_folder)
                    if os.path.isdir(inner_folder_path):
                        # 获取内层文件夹的范围
                        inner_start, inner_end = map(int, inner_folder.split('_'))
                        if inner_start <= file_number <= inner_end:
                            f2 = inner_folder_path.split("\\")[-1]
                            # 构造最终路径
                            # return os.path.join(inner_folder_path, filename)
                            return f1 + "/" + f2

    return None  # 如果没有找到合适的文件夹


# 定义文件保存路径
# TRANSUPLOAD_DIRECTORY = "C:/Users/86132/Desktop/28P-46PV3DRAW"
# # 确保 upload 目录存在
# os.makedirs(TRANSUPLOAD_DIRECTORY, exist_ok=True)
@app.post('/api/singleConvert/')  # 注意 API 路径前面需要加斜杠
async def upload_singlefile_convert(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 第一步，确定上传路径
    uploadbase = "/mnt/nfs/hndb/V3DRAW_16bit"  # 16bit的根目录
    result_path = find_storage_path(uploadbase, file.filename)
    parts = result_path.split('//')

    # 然后从第二部分中提取所需的内容
    result_path = parts[1].split('/')[-2] + '/' + parts[1].split('/')[-1]
    print("1result path", result_path)
    tmp = os.path.join(uploadbase, result_path).replace("\\", "/")
    print(tmp)
    print(file.filename)
    neuronImage = os.path.join(tmp, file.filename).replace("\\", "/")
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

    dbmip1 = os.path.join("MIP_Downsample", result_path).replace("\\", "/")
    dbmip2 = os.path.join(dbmip1, image.replace(".v3draw", '.tif')).replace("\\", "/")
    # stmt = (
    #     update(tracking_table)
    #         .where(tracking_table.c['Cell ID'] == image[:5])  # 这里假设表中有一个主键id列
    #         .values(image_file=dbmip2)
    # )
    # session.execute(stmt)
    db.query(models.HumanSingleCellTrackingTable) \
        .filter(models.HumanSingleCellTrackingTable.cell_id == image[:5]) \
        .update({models.HumanSingleCellTrackingTable.image_file: dbmip2})
    # 提交更改到数据库
    db.commit()

    mipbase = r"/mnt/nfs/hndb/MIP_Downsample"
    directory1 = os.path.join(mipbase, result_path).replace("\\", "/")

    outImage = os.path.join(directory1, image.replace(".v3draw", '.tif')).replace("\\", "/")
    # 创建文件
    with open(outImage, 'w') as file:
        pass  # 不执行任何操作
    pp = '1:1:e'

    cmd = f'xvfb-run -a -s "-screen 0 640x480x16" "/vaa3d/Vaa3D-x.1.1.4Ubuntu/Vaa3D-x" -x mipZSlices -f mip_zslices -i {neuronImage} -o {outImage} -p 1:1:e'
    os.system(cmd)

    # 转8bit
    bit8base = r"/mnt/nfs/hndb/V3DRAW_8bit"
    directory2 = os.path.join(bit8base, result_path).replace("\\", "/")
    outImage = os.path.join(directory2, '8bit_' + image).replace("\\", "/")
    # 创建文件
    with open(outImage, 'w') as file:
        pass  # 不执行任何操作

    cmd1 = f'xvfb-run -a -s "-screen 0 640x480x16" "/vaa3d/Vaa3D-x.1.1.4Ubuntu/Vaa3D-x" -x datatypeconvert -f dtc -i {neuronImage} -o {outImage} -p 1'
    os.system(cmd1)

    # 转pbd

    dbpbd1 = os.path.join("V3DPBD", result_path).replace("\\", "/")
    dbpbd2 = os.path.join(dbpbd1, image.replace(".v3draw", '.v3dpbd')).replace("\\", "/")
    # stmt = (
    #     update(tracking_table)
    #         .where(tracking_table.c['Cell ID'] == image[:5])  # 这里假设表中有一个主键id列
    #         .values(v3dpbd_file=dbpbd2)
    # )
    # session.execute(stmt)

    # session.commit()
    # # 关闭会话
    # session.close()
    db.query(models.HumanSingleCellTrackingTable) \
        .filter(models.HumanSingleCellTrackingTable.cell_id == image[:5]) \
        .update({models.HumanSingleCellTrackingTable.v3dpbd_file: dbpbd2})
    # 提交更改到数据库
    db.commit()

    pbdbase = r"/mnt/nfs/hndb/V3DPBD"
    directory3 = os.path.join(pbdbase, result_path).replace("\\", "/")
    outImage = os.path.join(directory3, image.replace(".v3draw", '.v3dpbd')).replace("\\", "/")
    # 创建文件
    with open(outImage, 'w') as file:
        pass  # 不执行任何操作
    cmd2 = f'xvfb-run -a -s "-screen 0 640x480x16" "/vaa3d/Vaa3D-x.1.1.4Ubuntu/Vaa3D-x" -x convert_file_format -f convert_format -i {neuronImage} -o {outImage}'

    os.system(cmd2)

    # 重命名
    new_file_name = tmp + "/" + image[:5] + ".v3draw"
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
    UPLOAD_DIR = "/mnt/nfs/hndb/2D_raw_images"
    # 第一步，确定上传路径
    result_path = find_storage_path(UPLOAD_DIR, folderName)
    parts = result_path.split('//')

    # 然后从第二部分中提取所需的内容
    result_path = parts[1].split('/')[-2] + '/' + parts[1].split('/')[-1]
    print("1result path", result_path)
    tmp = os.path.join(UPLOAD_DIR, result_path).replace("\\", "/")  # tmp是/mnt下的目录
    print(tmp)
    UPLOAD_DIR = os.path.join(tmp, folderName).replace("\\", "/")

    # 创建上传目录
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_files = []
    for file in files:
        file_location = os.path.join(UPLOAD_DIR, os.path.basename(file.filename)).replace("\\", '/')
        with open(file_location, "wb") as f:
            f.write(await file.read())
        saved_files.append(file_location)

    return JSONResponse(content={"uploaded_files": saved_files})


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
    access_token_expires = timedelta(minutes=60 * 24)  # 设置 token 有效期为 1 天
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


# 下载V3DPBD
@app.get("/api/download")
def download_file(file_path: str, cell_id: str, db: Session = Depends(get_db)):
    # 提取文件名和路径
    if len(file_path) == 5:
        PBDuploadbase = r"/mnt/nfs/hndb/V3DPBD"
        pbdpath = foundPBD.found_pbd_file(PBDuploadbase, file_path)  # 新的相对路径
        file_path = '/mnt/nfs/hndb/' + pbdpath
    else:
        file_path = '/mnt/nfs/hndb/' + file_path

    # 提取文件名和路径
    file_name = os.path.basename(file_path)
    directory = os.path.dirname(file_path)
    print(
        f"Request to download file: {file_path} with cell_id: {cell_id} \nfile_name: {file_name} \ndirectory: {directory} \n")

    # 从数据库获取数据
    cell_data = crud.get_single_cell_data_by_id(db, cell_id)
    print(f"Extracted cell_id: {cell_id}, cell_data: {cell_data}")
    print(f"soma_x, soma_y, soma_z: {cell_data.soma_x}, {cell_data.soma_y}, {cell_data.soma_z} \n")

    if not cell_data:
        raise HTTPException(status_code=404, detail="Cell data not found")

    if (cell_data.soma_x == '-' or cell_data.soma_y == '-' or cell_data.soma_z == '-') or \
            (cell_data.soma_x == '--' or cell_data.soma_y == '--' or cell_data.soma_z == '--'):
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
            return FileResponse(path=zip_file_path, filename=os.path.basename(zip_file_path),
                                media_type='application/zip')

        # 生成 .marker 文件
        marker_file_path = os.path.join('/mnt/nfs/hndb/temp', f"{cell_id}.marker")
        with open(marker_file_path, 'w') as marker_file:
            marker_file.write("##x,y,z,radius,shape,name,comment,color_r,color_g,color_b\n")
            marker_file.write(
                f"{cell_data.soma_x},{cell_data.soma_y},{cell_data.soma_z},0,0,{cell_data.cell_id},0,255,0,0\n")

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
        file_path = '/mnt/nfs/hndb/' + file_path
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
        small_number: Optional[str] = Query(None),
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
        'small_number': small_number.split(',') if small_number else [],
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
    sample_id_options = db.query(models.HumanSingleCellTrackingTable.patient_number).distinct().order_by(
        asc(models.HumanSingleCellTrackingTable.patient_number)).all()
    tissue_id_options = db.query(models.HumanSingleCellTrackingTable.tissue_block_number).distinct().order_by(
        asc(models.HumanSingleCellTrackingTable.tissue_block_number)).all()
    small_id_options = db.query(models.HumanSingleCellTrackingTable.small_number).distinct().order_by(
        asc(models.HumanSingleCellTrackingTable.small_number)).all()
    slice_id_options = db.query(models.HumanSingleCellTrackingTable.slice_number).distinct().order_by(
        asc(models.HumanSingleCellTrackingTable.slice_number)).all()
    brain_region_options = db.query(models.HumanSingleCellTrackingTable.brain_region).distinct().order_by(
        asc(models.HumanSingleCellTrackingTable.brain_region)).all()

    return {
        "sample_id_options": [{"value": option[0], "label": option[0]} for option in sample_id_options],
        "tissue_id_options": [{"value": option[0], "label": option[0]} for option in tissue_id_options],
        "small_id_options": [{"value": option[0], "label": option[0]} for option in small_id_options],
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
    "experiment_temperature", "experiment_humidity", "inject_method", "perfusion_date", "perfusion_staff",
    "ihc_category"
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
def create_single_cell_data(data: schemas.HumanSingleCellTrackingTableCreate, Authorize: AuthJWT = Depends(),
                            db: Session = Depends(get_db)):
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
        # sorted_details = sort_dict_by_order(data.dict(), field_order)
        # details = details = json.dumps(sorted_details)
        crud.create_user_log(db, int(user_id), f"Create single cell data with id {single_cell_data.cell_id}")

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
        # sorted_details = sort_dict_by_order(deleted_data_details, field_order)
        # details = json.dumps(sorted_details)

        # 执行删除操作
        deleted_data = crud.delete_single_cell_data(db=db, cell_id=cell_id)

        # 创建删除日志，记录详细信息
        crud.create_user_log(db, int(user_id), f"Delete single cell data with id {cell_id}")
        return deleted_data
    except Exception as e:
        logging.error(f"Error deleting single cell data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 编辑数据
@app.put("/api/singlecell/{cell_id}", response_model=schemas.HumanSingleCellTrackingTable)
def update_single_cell_data(cell_id: str, data: schemas.HumanSingleCellTrackingTableCreate,
                            Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    try:
        Authorize.jwt_required()
        user_id = Authorize.get_jwt_subject()  # 确保 user_id 是整数类型
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

        # details = json.dumps(changes)
        crud.create_user_log(db, int(user_id), f"Update single cell data with id {cell_id}")
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

from sqlalchemy import func, cast, Date
@app.get("/api/productiontrend/", response_model=dict)
def get_production_trend(start_date: str = None, end_date: str = None, db: Session = Depends(get_db)):
    try:
        # 构造基础查询
        query = db.query(
            func.date(models.HumanSingleCellTrackingTable.shooting_date).label('shooting_day'),
            func.count(models.HumanSingleCellTrackingTable.id).label('count_per_day')
        )

        # 根据日期范围过滤
        if start_date and end_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            query = query.filter(
                models.HumanSingleCellTrackingTable.shooting_date >= start_date_obj,
                models.HumanSingleCellTrackingTable.shooting_date <= end_date_obj
            )

        # 分组查询
        query = query.group_by(func.date(models.HumanSingleCellTrackingTable.shooting_date))
        # 排序按日期
        query = query.order_by(func.date(models.HumanSingleCellTrackingTable.shooting_date).asc())

        rows = query.all()

        # rows 形如: [(datetime.date(2023,1,1), 10), (datetime.date(2023,1,2), 12), ...]
        # 将结果拆分
        dates = [r[0].strftime('%Y-%m-%d') for r in rows]
        daily_values = [r[1] for r in rows]

        # 计算 totalValues：前缀和
        total_values = []
        running_sum = 0
        for val in daily_values:
            running_sum += val
            total_values.append(running_sum)

        # 通过聚合函数一次性获取min_date, max_date
        min_date_str = db.query(func.min(models.HumanSingleCellTrackingTable.shooting_date)).scalar()
        max_date_str = db.query(func.max(models.HumanSingleCellTrackingTable.shooting_date)).scalar()
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
    ).filter(
        models.HumanSingleCellTrackingTable.brain_region != '-',  # 排除 '-'
        models.HumanSingleCellTrackingTable.brain_region != '--',  # 排除 '--'
        models.HumanSingleCellTrackingTable.brain_region != '',  # 排除空白
    ).group_by(models.HumanSingleCellTrackingTable.brain_region).order_by(
        func.count(models.HumanSingleCellTrackingTable.id).desc()).all()
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
    # Query to get unique sample_id and patient_number where patient_number is valid
    data = db.query(models.Sample_Information.sample_id, models.Sample_Information.patient_number) \
        .filter(models.Sample_Information.patient_number != '--',
                models.Sample_Information.patient_number != '-',
                models.Sample_Information.patient_number != ''
                ).distinct().all()

    source_count = {}
    source_patient_numbers = {}  # Dictionary to track patient_numbers for each source
    unique_patient_numbers = set()  # Set to track unique patient numbers

    for item in data:
        sample_id, patient_number = item
        if patient_number not in unique_patient_numbers:
            unique_patient_numbers.add(patient_number)
            source = '-'.join(sample_id.split('-')[:2])  # Extract content before the second dash
            if source not in source_count:
                source_count[source] = 0
                source_patient_numbers[source] = []  # Initialize a list for each new source
            source_count[source] += 1
            source_patient_numbers[source].append(patient_number)  # Add the patient_number to the corresponding source

    return {
        "categories": list(source_count.keys()),  # List of unique sources
        "data": list(source_count.values()),       # List of corresponding counts
        "source_patient_numbers": source_patient_numbers
    }

@app.get("/api/sample-source-details")
def get_sample_source_details(db: Session = Depends(get_db)):
    """
    只进行2次主要查询:
    1) 获取 (sample_id, patient_number) 过滤掉无效值
    2) 获取 (patient_number, brain_region) 全表或带过滤
    """

    # 1) 获取有效的 sample_id, patient_number
    sample_data = (
        db.query(models.Sample_Information.sample_id, models.Sample_Information.patient_number)
        .filter(
            models.Sample_Information.patient_number.notin_(['--', '-', ''])
        )
        .distinct()
        .all()
    )

    # 映射 patient_number -> source
    # 另外统计 source_count, source_patient_numbers
    source_count = {}
    source_patient_numbers = {}
    unique_patient_numbers = set()

    for sample_id, patient_number in sample_data:
        if patient_number not in unique_patient_numbers:
            unique_patient_numbers.add(patient_number)
            source = "-".join(sample_id.split("-")[:2])
            if source not in source_count:
                source_count[source] = 0
                source_patient_numbers[source] = []
            source_count[source] += 1
            source_patient_numbers[source].append(patient_number)

    # 2) 获取 HumanSingleCellTrackingTable 的 (patient_number, brain_region) 用于统计
    tracking_data = (
        db.query(models.HumanSingleCellTrackingTable.patient_number, models.HumanSingleCellTrackingTable.brain_region)
        # 如果要只统计 patient_number 在 unique_patient_numbers 内的:
        # .filter(models.HumanSingleCellTrackingTable.patient_number.in_(unique_patient_numbers))
        .all()
    )

    # 将 tracking_data 组织成 dict: patient_number -> [brain_region, brain_region, ...]
    from collections import defaultdict
    patient_brain_regions = defaultdict(list)
    for (pt_num, br) in tracking_data:
        patient_brain_regions[pt_num].append(br)

    # 3) 根据 patient_brain_regions 计算 source_cell_count, source_brain_region_distribution
    source_cell_count = {}
    source_brain_region_distribution = {}
    for source in source_count.keys():
        source_cell_count[source] = 0
        source_brain_region_distribution[source] = {}

    # 遍历每个 source 下的所有 patient_number, 聚合
    for source, pt_list in source_patient_numbers.items():
        for pt_num in pt_list:
            # patient_brain_regions[pt_num] 可能为空或不存在
            br_list = patient_brain_regions.get(pt_num, [])
            source_cell_count[source] += len(br_list)
            for region in br_list:
                if region not in source_brain_region_distribution[source]:
                    source_brain_region_distribution[source][region] = 0
                source_brain_region_distribution[source][region] += 1

    return {
        "categories": list(source_count.keys()),
        "data": list(source_count.values()),
        "source_cell_count": source_cell_count,
        "source_brain_region_distribution": source_brain_region_distribution,
    }


# @app.get("/api/sample-source-details")
# def get_sample_source_details(db: Session = Depends(get_db)):
#     # Query to get unique sample_id and patient_number where patient_number is not '--'
#     data = db.query(models.Sample_Information.sample_id, models.Sample_Information.patient_number) \
#         .filter(models.Sample_Information.patient_number != '--',
#                 models.Sample_Information.patient_number != '-',
#                 models.Sample_Information.patient_number != ''
#                 ).distinct().all()

#     source_count = {}
#     source_patient_numbers = {}  # Dictionary to track patient_numbers for each source
#     source_cell_count = {}  # Dictionary to track HumanSingleCellTrackingTable counts for each source
#     source_brain_region_distribution = {}  # Dictionary to track brain region distribution for each source
#     unique_patient_numbers = set()  # Set to track unique patient numbers

#     for item in data:
#         sample_id, patient_number = item
#         if patient_number not in unique_patient_numbers:
#             unique_patient_numbers.add(patient_number)
#             source = '-'.join(sample_id.split('-')[:2])  # Extract content before the second dash
#             if source not in source_count:
#                 source_count[source] = 0
#                 source_patient_numbers[source] = []  # Initialize a list for each new source
#                 source_cell_count[source] = 0  # Initialize the count for tracking table data
#                 source_brain_region_distribution[source] = {}  # Initialize the brain region distribution
#             source_count[source] += 1
#             source_patient_numbers[source].append(patient_number)  # Add the patient_number to the corresponding source

#     # Query HumanSingleCellTrackingTable and count the entries for each patient_number
#     patient_cell_count = {}
#     for source, patient_numbers in source_patient_numbers.items():
#         for patient_number in patient_numbers:
#             # Query to get brain region distribution
#             tracking_data = db.query(models.HumanSingleCellTrackingTable.brain_region) \
#                 .filter(models.HumanSingleCellTrackingTable.patient_number == patient_number) \
#                 .all()

#             tracking_count = len(tracking_data)
#             patient_cell_count[patient_number] = tracking_count
#             source_cell_count[source] += tracking_count  # Add to the corresponding source's total tracking count

#             # Update brain region distribution
#             for brain_region in tracking_data:
#                 region = brain_region[0]  # Extract the brain_region from the query result
#                 if region not in source_brain_region_distribution[source]:
#                     source_brain_region_distribution[source][region] = 0
#                 source_brain_region_distribution[source][region] += 1

#     return {
#         "categories": list(source_count.keys()),
#         "data": list(source_count.values()),
#         "source_patient_numbers": source_patient_numbers,  # Include the patient number mapping
#         "patient_cell_count": patient_cell_count,  # Count of tracking table entries per patient number
#         "source_cell_count": source_cell_count,  # Count of tracking table entries per source
#         "source_brain_region_distribution": source_brain_region_distribution  # Brain region distribution per source
#     }

@app.get("/api/recons-distribution")
def get_recons_distribution(db: Session = Depends(get_db)):
    """
    统计 HumanSingleCellTrackingTable 中 swc_auto14 是否有值：
      - 已重建(有值)
      - 未重建(无值)
    返回形如: [ { name: '已重建', value: xxx }, { name: '未重建', value: yyy } ]
    """
    # 计算已重建
    reconstructed_count = db.query(models.HumanSingleCellTrackingTable) \
        .filter(models.HumanSingleCellTrackingTable.swc_auto14.isnot(None)) \
        .count()

    # 计算未重建
    not_reconstructed_count = db.query(models.HumanSingleCellTrackingTable) \
        .filter(models.HumanSingleCellTrackingTable.swc_auto14.is_(None)) \
        .count()

    return [
        {"name": "已重建", "value": reconstructed_count},
        {"name": "未重建", "value": not_reconstructed_count}
    ]


@app.post("/api/savereport/")
def save_report(report: schemas.Report, db: Session = Depends(get_db)):
    db_report = DailyReport(report_date=report.report_date, content=report.content)
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return {"msg": "Report saved successfully"}


@app.get("/api/latestreport/")
def get_latest_report(db: Session = Depends(get_db)):
    latest_report = db.query(DailyReport).order_by(desc(DailyReport.created_at)).first()
    if not latest_report:
        raise HTTPException(status_code=404, detail="No report found")
    return {
        "report_date": latest_report.report_date,
        "content": latest_report.content
    }


@app.get("/api/get-patientID")
def get_PIDoptions(db: Session = Depends(get_db)):
    sample_id_options = db.query(models.Sample_Information.patient_number).filter(
        models.Sample_Information.patient_number != '',
        models.Sample_Information.patient_number != '-',
        models.Sample_Information.patient_number != '--'
    ).distinct().order_by(asc(models.Sample_Information.patient_number)).all()
    return {
        "pid_options": [{"value": option[0], "label": option[0]} for option in sample_id_options]
    }

@app.get("/api/sample_information/", response_model=dict)
def read_sample_information(
    skip: int = 0,
    limit: int = 20,
    sample_source: List[str] = Query(None),  # 接收多个sample_source
    PID: List[str] = Query(None),            # 接收多个PID
    db: Session = Depends(get_db)
):
    """
    sample_source 会是一个字符串列表，
    比如 ["BJ-TT", "NanJ-JZ"]。
    PID 同理。
    """
    print('Sample Source:', sample_source)
    print('PID:', PID)

    query = db.query(models.Sample_Information)

    # 根据 sample_source 过滤: 例如 OR 逻辑
    if sample_source:
        # 如果后端业务逻辑是：来源在这些值里之一即可
        # 常见写法： (sample_id LIKE 'BJ-TT-%') OR (sample_id LIKE 'NanJ-JZ-%')
        from sqlalchemy import or_
        or_clauses = []
        for src in sample_source:
            if src != 'none':  # 或者先判断是否要排除 'none'
                or_clauses.append(
                    models.Sample_Information.sample_id.like(f"{src}-%")
                )
        if or_clauses:
            query = query.filter(or_(*or_clauses))

    # 根据 PID 过滤: 例如 OR 逻辑
    if PID:
        from sqlalchemy import or_
        or_clauses_pid = []
        for pid_val in PID:
            if pid_val != 'none':
                or_clauses_pid.append(
                    models.Sample_Information.patient_number == pid_val
                )
        if or_clauses_pid:
            query = query.filter(or_(*or_clauses_pid))

    # 排序
    query = query.order_by(cast(models.Sample_Information.total_id, Integer))

    total = query.count()
    data = query.offset(skip).limit(limit).all()

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
    "idx", "total_id", "patient_id", "sample_id", "id", "patient_number", "tissue_id",
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
        db_sample.sample_snapshot = ""
        db_sample.sample_image = ""
        db_sample.sample_annotation = ""
        db.add(db_sample)
        db.commit()
        db.refresh(db_sample)

        sorted_details = sort_dict_by_order(sample.dict(), sample_field_order)
        # details = json.dumps(sorted_details, ensure_ascii=False)  # ensure_ascii=False 用来正确处理中文内容，确保中文字符不被转义
        crud.create_user_log(db, int(user_id), f"Create sample information with idx {db_sample.idx}")

        return db_sample
    except Exception as e:
        logging.error(f"Error creating sample information: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 编辑
@app.put("/api/sample_information/{idx}", response_model=schemas.SampleInfo)
def update_sample_information(idx: int, updated_info: schemas.SampleInfoCreate, Authorize: AuthJWT = Depends(),
                              db: Session = Depends(get_db)):
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
        # details = json.dumps(changes, ensure_ascii=False)
        crud.create_user_log(db, int(user_id), f"Update sample information with idx {idx}")

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
        # details = json.dumps(sorted_details, ensure_ascii=False)

        db.delete(sample_info)
        db.commit()

        crud.create_user_log(db, int(user_id), f"Delete sample information with idx {idx}")
        return sample_info
    except Exception as e:
        logging.error(f"Error deleting sample information: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 记录本照片上传路径
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


# 灌注文件上传路径
# ORIGINAL_UPLOAD_DIR = "Injection_Files/Original"
# TEMP_DIR = "Injection_Files/temp"
# DB_UPLOAD_DIR = "Injection_Files/DB_Uploads"  # 设置CSV文件上传的目录

ORIGINAL_UPLOAD_DIR = "/mnt/nfs/hndb/Injection_Files/Original"
TEMP_DIR = "/mnt/nfs/hndb/Injection_Files/temp"
DB_UPLOAD_DIR = "/mnt/nfs/hndb/Injection_Files/DB_Uploads"  # 设置CSV文件上传的目录


# 存储多个文件到子文件夹
@app.post("/api/upload_files")
async def upload_files(subfolder_name: str = Form(...), files: List[UploadFile] = File(...)):
    # 创建子文件夹
    subfolder_path = os.path.join(ORIGINAL_UPLOAD_DIR, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # 处理并保存上传的文件
    for file in files:
        # 获取文件扩展名
        file_extension = os.path.splitext(file.filename)[1].lower()
        # 如果是 .csv 文件，重命名为子文件夹名称
        if file_extension == '.csv':
            file_path = os.path.join(subfolder_path, f"{subfolder_name}.csv")
        else:
            file_path = os.path.join(subfolder_path, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    return {"message": "Files stored successfully"}


# 检查文件是否存在
@app.get("/api/check_csv_exists")
async def check_csv_exists(filename: str):
    file_path = os.path.join(DB_UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return {"exists": True}
    return {"exists": False}


REQUIRED_COLUMNS = [
    "Id", "sample_preparation_date", "sample_preparation_time", "sample_preparation_staff", "slice_thickness",
    "fresh_perfusion", "Status", "dye_name", "dye_concentration(%)", "immunohistochemistry",
    "primaryAntibody_concentration", "secondAntibody_band", "DAPI_concentration", "perfusion_date",
    "perfusion_time", "AddingTime", "Depth", "current_intensity", "perfusion_time_on", "perfusion_time_off",
    "experiment_temperature", "experiment_humidity", "perfusion_user", "X", "Y", "Z", "AddingX", "AddingY", "AddingZ"
]


# def convert_to_iso_format(date_str):
#     """Attempt to convert various date formats to YYYY-MM-DD."""
#     formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"]
#     for fmt in formats:
#         try:
#             return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
#         except ValueError:
#             continue
#     # If no format matched, raise an error
#     raise ValueError("Unable to convert date format")

@app.post("/api/upload_csv_to_db")
async def upload_csv_to_db(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 确保目录存在
    os.makedirs(DB_UPLOAD_DIR, exist_ok=True)

    # 准备保存文件的路径
    file_path = os.path.join(DB_UPLOAD_DIR, file.filename)

    # 将文件内容保存到内存中，以便后续操作
    file_content = file.file.read()

    # 解析文件名
    file_name_without_ext = file.filename.rsplit('.', 1)[0]
    pattern = re.match(r"^(P\d{5})-(T\d{3})-(R\d{3})-(S\d{3})(?:-(B\d))?$", file_name_without_ext)
    if not pattern:
        raise HTTPException(status_code=400, detail="Invalid file name format. Please check the format and try again.")

    # 提取 P, T, R, S, (B) 部分
    p_part, t_part, r_part, s_part, b_part = pattern.groups()
    prefix = f"{p_part}_{t_part}_{r_part}_{s_part}"
    if b_part:
        prefix += f"_{b_part}"
    prefix_pattern = re.compile(rf"^{re.escape(prefix)}_C\d+$")

    try:
        # 读取 CSV 文件内容并将其转换为 DataFrame
        df = pd.read_csv(pd.io.common.BytesIO(file_content), encoding='utf-8')

        # 检查 CSV 文件中是否有 ID 列
        if 'Id' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain an ID column.")

        # 检查 ID 列中的所有值是否符合文件名中的格式
        if not df['Id'].apply(lambda x: bool(prefix_pattern.match(str(x)))).all():
            raise HTTPException(status_code=400, detail="File name does not match its ID column.")

        # 新增检查 2：C 编号是否有重复
        c_numbers = df['Id'].apply(lambda x: re.search(r"C\d{5}$", str(x)).group())
        if c_numbers.duplicated().any():
            raise HTTPException(status_code=400,
                                detail="Duplicate C numbers found in ID column. Please check and re-upload.")

        # 检查所有必需列是否存在
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")

        # 检查必需列的空值（perfusion_time 和 AddingTime 除外）
        empty_columns = [col for col in REQUIRED_COLUMNS if
                         col not in ["perfusion_time", "AddingTime"] and df[col].isnull().any()]
        if empty_columns:
            raise HTTPException(status_code=400, detail=f"Columns with missing values: {', '.join(empty_columns)}")

        # 从文件名中提取数值
        p_number = int(p_part[1:])  # 取出 P 部分的数值
        t_number = int(t_part[1:])  # 取出 T 部分的数值

        # 查询数据库，验证 P 和 T 编号是否存在，只比较数值部分
        sample_info = db.query(models.Sample_Information).filter(
            func.cast(func.substr(models.Sample_Information.patient_number, 2), Integer) == p_number,  # 去掉 "P" 并只比较数值
            func.cast(func.substr(models.Sample_Information.tissue_id, 2), Integer) == t_number  # 去掉 "T" 并只比较数值
        ).first()

        if not sample_info:
            raise HTTPException(status_code=400, detail="No matching sample found.")

        # # 新增：检查 dye_name 列是否包含 '-1'
        # if df['dye_name'].astype(str).str.strip().eq('-1').any():
        #     raise HTTPException(status_code=400, detail="Abnormal value in dye_name column.")

        # 新增：检查 dye_name 列是否包含数值或数值型字符串
        def is_numeric(value):
            try:
                # 尝试将值转换为浮点数
                float(str(value).strip())
                return True
            except ValueError:
                return False

        if df['dye_name'].apply(is_numeric).any():
            raise HTTPException(status_code=400, detail="Abnormal value in dye_name column.")

        # Check concentration columns for integer-only values
        concentration_columns = ['primaryAntibody_concentration', 'DAPI_concentration']
        for col in concentration_columns:
            if not df[col].apply(lambda x: isinstance(x, int) or str(x).isdigit() or str(x) == '-').all():
                raise HTTPException(status_code=400, detail="Concentration contents error.")

        # 将日期列转换为 datetime 对象
        date_columns = ['sample_preparation_date', 'perfusion_date']
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='raise', infer_datetime_format=True)
            except ValueError as e:
                raise HTTPException(status_code=400,
                                    detail=f"Unable to convert date format in column {date_col}: {str(e)}")

        ## 添加 ihc_category 列并根据 dye_name 列设置值
        # if 'ihc_category' not in df.columns:
        #     df['ihc_category'] = df['dye_name'].apply(lambda x: 'Lucifer Yellow' if x == 'Lucifer Yellow' else '-')
        # else:
        #     df.loc[df['dye_name'] == 'Lucifer Yellow', 'ihc_category'] = 'Lucifer Yellow'
        #     df.loc[df['dye_name'] != 'Lucifer Yellow', 'ihc_category'] = '-'

        # 修改：处理 ihc_category 列
        cutoff_date = pd.to_datetime('2024-10-29')

        def process_ihc_category(row):
            if row['perfusion_date'] <= cutoff_date:
                # 现有逻辑
                if row['dye_name'] == 'Lucifer Yellow':
                    return 'Lucifer Yellow'
                else:
                    return '-'
            else:
                # 保留原始值，不进行处理
                return row['ihc_category']

        df['ihc_category'] = df.apply(process_ihc_category, axis=1)

        df['sample_preparation_date'] = df['sample_preparation_date'].dt.strftime('%Y-%m-%d')
        df['perfusion_date'] = df['perfusion_date'].dt.strftime('%Y-%m-%d')

        # 添加一个新列 file_name 并将所有行的值设置为当前文件名
        df['file_name'] = file.filename

        df = df.replace({np.nan: '--'})

        # 尝试将 DataFrame 插入数据库
        table = Table('injection_table_20241028', MetaData(), autoload_with=db.bind)
        try:
            for _, row in df.iterrows():
                # print(row)
                stmt = insert(table).values(row.to_dict())
                db.execute(stmt)

            db.commit()

        except SQLAlchemyError as db_error:
            print(f"Database insertion failed: {db_error}")
            raise HTTPException(status_code=400, detail=f"Database insertion failed: {str(db_error)}")

        # try:
        #     df.to_sql('injection_table_20241010', con=db.bind, if_exists='append', index=False)

        # except Exception as db_error:
        #     raise HTTPException(status_code=400, detail=f"Database insertion failed: {str(db_error)}")

        # 如果数据库插入成功，保存文件到服务器
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")

    return {"message": "CSV uploaded and stored in the database successfully"}


# 列出子文件夹(未检查的文件夹列表）
@app.get("/api/folders")
async def list_folders():
    # 获取 ORIGINAL_UPLOAD_DIR 中的所有子文件夹
    folders = [f for f in os.listdir(ORIGINAL_UPLOAD_DIR) if os.path.isdir(os.path.join(ORIGINAL_UPLOAD_DIR, f))]

    # 过滤出 DB_UPLOAD_DIR 中没有对应 CSV 文件的子文件夹
    filtered_folders = []
    for folder in folders:
        # 构建对应的 CSV 文件名
        csv_file_name = f"{folder}.csv"
        csv_file_path = os.path.join(DB_UPLOAD_DIR, csv_file_name)

        # 检查 CSV 文件是否存在
        if not os.path.exists(csv_file_path):
            filtered_folders.append(folder)

    return {"folders": filtered_folders}


# 下载子文件夹
@app.get("/api/download_folder")
async def download_folder(folder: str):
    folder_path = os.path.join(ORIGINAL_UPLOAD_DIR, folder)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    # 确保临时目录存在
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 在临时目录中创建 zip 文件
    zip_file_path = os.path.join(TEMP_DIR, f"{folder}.zip")
    shutil.make_archive(os.path.join(TEMP_DIR, folder), 'zip', folder_path)

    return FileResponse(zip_file_path, media_type='application/zip', filename=f"{folder}.zip")


# 定义文件保存路径
# IMAGING_METADATA_DIR = "Imaging_Files/Metadata"
# MARKER_FILES_DIR = "Imaging_Files/Markers"
# ANNOTATION_FILES_DIR = "Imaging_Files/Annotations"
IMAGING_METADATA_DIR = "/mnt/nfs/hndb/Imaging_Files/Metadata"
MARKER_FILES_DIR = "/mnt/nfs/hndb/Imaging_Files/Markers"
ANNOTATION_FILES_DIR = "/mnt/nfs/hndb/Imaging_Files/Annotations"
IMAGING_MATCHTABLE_DIR = "/mnt/nfs/hndb/Imaging_Files/MatchTables"

# 确保目录存在
os.makedirs(IMAGING_METADATA_DIR, exist_ok=True)
os.makedirs(MARKER_FILES_DIR, exist_ok=True)
os.makedirs(ANNOTATION_FILES_DIR, exist_ok=True)


# Helper functions to check if a string is an integer or float
def is_float(s):
    pattern = r'^[-+]?[0-9]*\.[0-9]+$'
    return bool(re.match(pattern, s))


def is_integer(s):
    return s.isdigit()


# Function to read marker file lines into a DataFrame
def read_marker_lines(lines):
    result_dict = {}
    headers = []
    row_count = 0
    for line in lines:
        # 跳过空白行
        if not line.strip():
            continue
        if line.startswith('##'):
            if not result_dict:
                headers = [header.strip() for header in line[2:].strip().split(',')]
                row_count = len(headers)
                for header in headers:
                    result_dict[header] = []
            else:
                raise Exception('Duplicated headers in marker file!')
        else:
            if result_dict:
                cells = [cell.strip() for cell in line.strip().split(',')]
                if len(cells) != row_count:
                    raise Exception('Missing columns of data!')
                for i in range(row_count):
                    header = headers[i]
                    cell = cells[i]
                    if is_integer(cell):
                        cell = int(cell)
                    elif is_float(cell):
                        cell = float(cell)
                    result_dict[header].append(cell)
            else:
                raise Exception('Missing headers before data!')
    result = pd.DataFrame(result_dict)
    return result


# Updated upload_imaging_info API
@app.post("/api/upload_imaging_info")
async def upload_imaging_info(
        metadata_files: List[UploadFile] = File([]),
        marker_files: List[UploadFile] = File([]),
        annotation_files: List[UploadFile] = File([]),
        db: Session = Depends(get_db)  # 注入数据库会话
):
    uploaded_files = []

    # Function to validate sample numbers against the database
    def validate_sample_number(file_name: str):
        # 提取文件名中的 P 和 T 编号
        file_pattern = r"^P(\d{5})-T(\d{3})-R\d{3}-S\d{3}(-B\d)?(-\d+)?"
        match = re.match(file_pattern, file_name)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filename format for sample number check: {file_name}"
            )
        p_number, t_number = int(match.group(1)), int(match.group(2))

        # 查询数据库，验证 P 和 T 编号是否存在
        sample_info = db.query(models.Sample_Information).filter(
            func.cast(func.substr(models.Sample_Information.patient_number, 2), Integer) == p_number,
            func.cast(func.substr(models.Sample_Information.tissue_id, 2), Integer) == t_number
        ).first()

        if not sample_info:
            raise HTTPException(
                status_code=400,
                detail=f"No matching sample found for P{p_number} and T{t_number}. Please check the file: {file_name}"
            )

    # Process metadata files
    for file in metadata_files:
        try:
            # Validate filename format
            if not re.match(r"^P\d{5}-T\d{3}-R\d{3}-S\d{3}(-B\d)?(-\d+)?-[A-Z]{2,3}\.(xlsx|xml)$", file.filename):
                raise HTTPException(status_code=400,
                                    detail="Invalid filename format for metadata file. Expected format: P00001-T001-R001-S001(-B1)(-1)-NAME.xlsx or .xml")

            # Validate sample number in the file name
            validate_sample_number(file.filename)

            # Check if file already exists
            file_path = os.path.join(IMAGING_METADATA_DIR, file.filename)
            if os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"'{file.filename}' already exists. Please check.")

            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)
        except HTTPException as e:
            # Raise exception with uploaded_files
            raise HTTPException(status_code=400, detail={"error": e.detail, "uploaded_files": uploaded_files})
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": str(e), "uploaded_files": uploaded_files})

    # Process marker files
    for file in marker_files:
        try:
            # Validate filename and extract P, T, R, S, B
            file_pattern = r"^(P\d{5})-(T\d{3})-(R\d{3})-(S\d{3})(-B\d)?(-\d+)?\.marker$"
            match = re.match(file_pattern, file.filename)
            if not match:
                raise HTTPException(status_code=400,
                                    detail="Invalid filename format for marker file. Expected format: P00001-T001-R001-S001(-B1)(-1).marker")
            file_parts = match.groups()
            file_P = file_parts[0]
            file_T = file_parts[1]
            file_R = file_parts[2]
            file_S = file_parts[3]
            file_B = file_parts[4] if file_parts[4] else ''
            file_number = file_B.lstrip('-B') if file_B else ''

            # Validate sample number in the file name
            validate_sample_number(file.filename)

            # Read marker file content
            file.file.seek(0)
            lines = [line.decode('utf-8').strip() for line in file.file.readlines()]

            try:
                marker_df = read_marker_lines(lines)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading marker file '{file.filename}': {str(e)}")

            # Check required columns
            required_columns = {'name', 'x', 'y', 'z'}
            if not required_columns.issubset(marker_df.columns):
                raise HTTPException(status_code=400,
                                    detail=f"Marker file '{file.filename}' is missing required columns: {required_columns}")

            # Validate 'name' column entries
            for name_entry in marker_df['name']:
                # Add a type check for name_entry
                if not isinstance(name_entry, str):
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{file.filename}' File name and its 'name' column do not match."
                    )
                name_pattern = r"^(P\d{5})_(T\d{3})_(R\d{3})_(S\d{3})(_B\d)?_C\d+"
                name_match = re.match(name_pattern, name_entry)
                if not name_match:
                    raise HTTPException(status_code=400,
                                        detail=f"Invalid 'name' entry in marker file '{file.filename}': {name_entry}")
                name_parts = name_match.groups()
                name_P = name_parts[0]
                name_T = name_parts[1]
                name_R = name_parts[2]
                name_S = name_parts[3]
                name_B = name_parts[4] if name_parts[4] else ''
                name_number = name_B.lstrip('_B') if name_B else ''

                # Compare file_P with name_P, etc.
                if (
                        file_P != name_P or file_T != name_T or file_R != name_R or file_S != name_S or file_number != name_number):
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{file.filename}' File name and its 'name' column do not match. Please check."
                    )

            # Step 1: Check for duplicate 'C' identifiers
            if marker_df['name'].duplicated().any():
                duplicate_names = marker_df[marker_df['name'].duplicated()]['name'].unique()
                raise HTTPException(
                    status_code=400,
                    # detail=f"Duplicate 'C' identifiers found in marker file '{file.filename}': {', '.join(duplicate_names)}"
                    detail="Duplicate C numbers found in 'name' column. Please check."
                )

            # Step 2: Validate 'name' column format
            invalid_names = marker_df[~marker_df['name'].str.match(r'^P\d{5}_T\d{3}_R\d{3}_S\d{3}_B\d?_C\d{5}$', na=False)]
            if not invalid_names.empty:
                raise HTTPException(
                    status_code=400,
                    # detail=f"Invalid 'name' entries in marker file '{file.filename}': {', '.join(invalid_names['name'].unique())}"
                    detail="Invalid C number in the 'name' column. Please check."
                )

            # Check if file already exists
            file_path = os.path.join(MARKER_FILES_DIR, file.filename)
            if os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"'{file.filename}' already exists. Please check.")

            # Save file
            file.file.seek(0)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)
        except HTTPException as e:
            # Raise exception with uploaded_files
            raise HTTPException(status_code=400, detail={"error": e.detail, "uploaded_files": uploaded_files})
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": str(e), "uploaded_files": uploaded_files})

    # Process annotation files
    for file in annotation_files:
        try:
            # Validate filename format
            if not re.match(r"^P\d{5}-T\d{3}-R\d{3}-S\d{3}(-B\d)?(-\d+)?\.apo$", file.filename):
                raise HTTPException(status_code=400,
                                    detail="Invalid filename format for annotation file. Expected format: P00001-T001-R001-S001(-B1)(-1).apo")

            # Validate sample number in the file name
            validate_sample_number(file.filename)

            # Check if file already exists
            file_path = os.path.join(ANNOTATION_FILES_DIR, file.filename)
            if os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"'{file.filename}' already exists. Please check.")

            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)
        except HTTPException as e:
            raise HTTPException(status_code=400, detail={"error": e.detail, "uploaded_files": uploaded_files})
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": str(e), "uploaded_files": uploaded_files})

    return JSONResponse(content={"message": "Files uploaded successfully", "uploaded_files": uploaded_files})
@app.post("/api/upload_injection_file")
async def upload_injection_file(
    file: UploadFile = File(...),
    Authorize: AuthJWT = Depends(),
    db: Session = Depends(get_db)
):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    sample_preparation_id = os.path.splitext(file.filename)[0]
    upload_path = os.path.join("/mnt/nfs/hndb/SamplePreparation", sample_preparation_id)
    os.makedirs(upload_path, exist_ok=True)
    file_path = os.path.join(upload_path, file.filename)

    # Read file content asynchronously
    file_content = await file.read()

    # Parse filename
    file_name_without_ext = file.filename.rsplit('.', 1)[0]
    pattern = re.match(r"^(P\d{5})-(T\d{3})-(R\d{3})-(S\d{3})(?:-(B\d))?$", file_name_without_ext)
    if not pattern:
        raise HTTPException(status_code=400, detail="Invalid file name format. Please check the format and try again.")

    # Extract parts from filename
    p_part, t_part, r_part, s_part, b_part = pattern.groups()
    prefix = f"{p_part}_{t_part}_{r_part}_{s_part}"
    if b_part:
        prefix += f"_{b_part}"
    prefix_pattern = re.compile(rf"^{re.escape(prefix)}_C\d+$")

    try:
        # Read CSV content into DataFrame
        df = pd.read_csv(pd.io.common.BytesIO(file_content), encoding='utf-8')

        # Check if all required columns are present
        missing_columns = [col for col in REQUIRED_COLUMNS_NEW if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")

        # Check if 'Id' column exists
        if 'Id' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain an ID column.")

        # Validate 'Id' column against filename pattern
        if not df['Id'].apply(lambda x: bool(prefix_pattern.match(str(x)))).all():
            raise HTTPException(status_code=400, detail="File name does not match its ID column.")

        # Check for duplicate C numbers
        c_numbers = df['Id'].apply(lambda x: re.search(r"C\d{5}$", str(x)).group())
        if c_numbers.duplicated().any():
            raise HTTPException(status_code=400, detail="Duplicate C numbers found in ID column. Please check and re-upload.")

        # Check for missing values in required columns (excluding 'perfusion_time' and 'AddingTime')
        empty_columns = [col for col in REQUIRED_COLUMNS if
                         col not in ["perfusion_time", "AddingTime"] and df[col].isnull().any()]
        if empty_columns:
            raise HTTPException(status_code=400, detail=f"Columns with missing values: {', '.join(empty_columns)}")

        # Extract numerical parts from filename
        p_number = int(p_part[1:])  # Remove 'P' and convert to int
        t_number = int(t_part[1:])  # Remove 'T' and convert to int

        # Validate P and T numbers in the database
        sample_info = db.query(models.Sample_Information).filter(
            func.cast(func.substr(models.Sample_Information.patient_number, 2), Integer) == p_number,
            func.cast(func.substr(models.Sample_Information.tissue_id, 2), Integer) == t_number
        ).first()

        if not sample_info:
            raise HTTPException(status_code=400, detail="No matching sample found.")

        # Save the file to the specified path
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Process 'perfusion_date' and 'perfusion_user'
        perfusion_date = df['perfusion_date'].astype(str).iloc[0]
        perfusion_user = df['perfusion_user'].astype(str).iloc[0]

        # Process 'dye_name' to calculate unique dyes and combine their names
        unique_dyes = df['dye_name'].dropna().unique().tolist()
        dyes = len(unique_dyes)
        combined_dye_names = ','.join(unique_dyes)

        # Calculate injected_num (count of non-'missing' status records)
        injection_num = df[df['Status'] != 'Missing'].shape[0]

        # Calculate needles (unique values in needle_name column)
        if 'Needle_name' in df.columns:
            unique_needles = df['Needle_name'].dropna().unique().tolist()
            needles = len(unique_needles)
        else:
            needles = -1

        # Extract values from filename
        block_id = b_part if b_part else '--'

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")

    # Log the upload action
    details = json.dumps(file_path)
    crud.create_user_log(db, int(user_id), f"Upload injection_file of: {sample_preparation_id}",
                        details=details)

    # Prepare response data
    response_data = {
        "id": None,
        "sampleId": p_part,
        "tissueId": t_part,
        "rollId": r_part,
        "sliceId": s_part,
        "blockId": block_id,
        "dyes": dyes,
        "needles": needles,
        "status": "injected",
        "injected_num": injection_num,
        "perfusion_user": perfusion_user,
        "perfusion_date": perfusion_date,
        "dye_name": combined_dye_names,
    }

    return response_data

@app.post("/api/upload_imaging_annotation_file/{sample_preparation_id}/{imaging_id}")
async def upload_imaging_annotation_file(
    annotation_file: UploadFile = File(...),
    sample_preparation_id: str = '',
    imaging_id: str = '',
    Authorize: AuthJWT = Depends(),
    db: Session = Depends(get_db),
):
    """
    Upload an annotation file, validate it, and store it in the appropriate folder.
    Now includes CSV (like) header validation.
    """
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    uploaded_files = []

    def validate_sample_number(file_name: str):
        """
        Validate the sample number in the filename
        by extracting P and T from the filename and
        checking against the database.
        """
        # Example pattern: P00001-T001-R001-S001(-B1)(-1).apo
        file_pattern = r"^P(\d{5})-T(\d{3})-R\d{3}-S\d{3}(-B\d)?(-\d+)?"
        match = re.match(file_pattern, file_name)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filename format for sample number check: {file_name}"
            )
        p_number, t_number = int(match.group(1)), int(match.group(2))
        # print(p_number, t_number)  # for debugging

        # Query the database, verifying P and T exist
        sample_info = db.query(models.Sample_Information).filter(
            func.cast(func.substr(models.Sample_Information.patient_number, 2), Integer) == p_number,
            func.cast(func.substr(models.Sample_Information.tissue_id, 2), Integer) == t_number
        ).first()

        if not sample_info:
            raise HTTPException(
                status_code=400,
                detail=f"No matching sample found for P{p_number} and T{t_number}. Please check the file: {file_name}"
            )

    try:
        file = annotation_file
        # Validate filename format
        if not re.match(r"^P\d{5}-T\d{3}-R\d{3}-S\d{3}(-B\d)?(-\d+)?\.apo$", file.filename):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid filename format for annotation file. "
                    "Expected: P00001-T001-R001-S001(-B1)(-1).apo"
                )
            )

        # Validate sample number in the file name
        validate_sample_number(file.filename)

        # ─────────────────────────────────────────────────────────────────────────
        # STEP: Validate CSV-like structure for required columns
        # We'll read the entire file in memory. If your file is extremely large,
        # consider streaming or chunk-based approaches.
        # ─────────────────────────────────────────────────────────────────────────
        required_columns = [
            "orderinfo", "name", "comment",
            "z", "x", "y",
            "color_r", "color_g", "color_b"
        ]
        try:
            # Read entire file as text
            file_contents = file.file.read().decode("utf-8", errors="ignore")
            # Reset file pointer so we can save later, if needed
            file.file.seek(0)

            # Parse as CSV
            csv_reader = csv.DictReader(StringIO(file_contents))
            # Check if all required columns are in the CSV header
            if not csv_reader.fieldnames:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} has no header row or could not be parsed as CSV."
                )

            normalized_fieldnames = [name.strip() for name in csv_reader.fieldnames]
            # Replace original fieldnames with stripped versions
            csv_reader.fieldnames = normalized_fieldnames

            # Check for missing columns
            missing_columns = [
                col for col in required_columns
                if col not in csv_reader.fieldnames
            ]
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File {file.filename} is missing required columns: {missing_columns}. "
                        "Please check your file format."
                    )
                )
        except HTTPException as e:
            raise e  # re-raise the HTTPException
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not validate file as CSV. Error: {str(e)}"
            )

        # ─────────────────────────────────────────────────────────────────────────
        # STEP: Define the correct folder path and save file
        # ─────────────────────────────────────────────────────────────────────────
        if imaging_id == '--':
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}"
        else:
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}-{imaging_id}"
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, file.filename)

        # Uncomment if you need to prevent overwriting existing files:
        # if os.path.exists(file_path):
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"'{file.filename}' already exists. Please check."
        #     )

        # Save the file
        with open(file_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)

    except HTTPException as e:
        # Return partial information about already uploaded files
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.detail, "uploaded_files": uploaded_files}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "uploaded_files": uploaded_files}
        )

    # Log details
    details = json.dumps(file_path)
    crud.create_user_log(
        db,
        int(user_id),
        f"Upload imaging_annotation_file of: {sample_preparation_id}",
        details=details
    )
    return JSONResponse(
        content={
            "message": "File uploaded successfully",
            "uploaded_files": uploaded_files
        }
    )

@app.post("/api/upload_imaging_metadata/{sample_preparation_id}/{imaging_id}")
async def upload_imaging_metadata(
        metadata_file: UploadFile = File,
        sample_preparation_id: str = '',
        imaging_id: str = '',
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)  # 注入数据库会话
):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    uploaded_files = []

    # Function to validate sample numbers against the database
    def validate_sample_number(file_name: str):
        # 提取文件名中的 P 和 T 编号
        file_pattern = r"^P(\d{5})-T(\d{3})-R\d{3}-S\d{3}(-B\d)?(-\d+)?"
        match = re.match(file_pattern, file_name)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filename format for sample number check: {file_name}"
            )
        p_number, t_number = int(match.group(1)), int(match.group(2))
        print(p_number, t_number)

        # 查询数据库，验证 P 和 T 编号是否存在
        sample_info = db.query(models.Sample_Information).filter(
            func.cast(func.substr(models.Sample_Information.patient_number, 2), Integer) == p_number,
            func.cast(func.substr(models.Sample_Information.tissue_id, 2), Integer) == t_number
        ).first()

        if not sample_info:
            raise HTTPException(
                status_code=400,
                detail=f"No matching sample found for P{p_number} and T{t_number}. Please check the file: {file_name}"
            )

    # Process metadata files

    try:
        file = metadata_file
        # Validate filename format
        print('filename', file.filename)
        if not re.match(r"^P\d{5}-T\d{3}-R\d{3}-S\d{3}(-B\d)?(-\d+)?-[A-Za-z_]{2,10}\.(xlsx|xml)$", file.filename):
            raise HTTPException(status_code=400,
                                detail="Invalid filename format for metadata file. Expected format: P00001-T001-R001-S001(-B1)(-1)-NAME.xlsx or .xml")

        # Validate sample number in the file name
        print('1')
        validate_sample_number(file.filename)
        print('2')
        if imaging_id == '--':
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}"
        else:
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}-{imaging_id}"
        # Check if file already exists
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, file.filename)
        #放到前端做重复检查以及是否覆盖
        # if os.path.exists(file_path):
        #     raise HTTPException(status_code=400, detail=f"'{file.filename}' already exists. Please check.")

        # Save file
        with open(file_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    except HTTPException as e:
        # Raise exception with uploaded_files
        raise HTTPException(status_code=400, detail={"error": e.detail, "uploaded_files": uploaded_files})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "uploaded_files": uploaded_files})
    details = json.dumps(file_path)
    crud.create_user_log(db, int(user_id),
                         f"Upload imaging_metadata of: {sample_preparation_id}",
                         details=details)
    return JSONResponse(status_code=200,
                        content={"message": "File uploaded successfully", "uploaded_files": uploaded_files})


@app.post("/api/upload_imaging_marker/{sample_preparation_id}/{imaging_id}")
async def upload_imaging_marker(
        marker_file: UploadFile = File,
        Authorize: AuthJWT = Depends(),
        sample_preparation_id: str = '',
        imaging_id: str = '',
        db: Session = Depends(get_db)  # 注入数据库会话
):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    uploaded_files = []

    # Function to validate sample numbers against the database
    def validate_sample_number(file_name: str):
        # 提取文件名中的 P 和 T 编号
        file_pattern = r"^P(\d{5})-T(\d{3})-R\d{3}-S\d{3}(-B\d)?(-\d+)?"
        match = re.match(file_pattern, file_name)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filename format for sample number check: {file_name}"
            )
        p_number, t_number = int(match.group(1)), int(match.group(2))
        print(p_number, t_number)

        # 查询数据库，验证 P 和 T 编号是否存在
        sample_info = db.query(models.Sample_Information).filter(
            func.cast(func.substr(models.Sample_Information.patient_number, 2), Integer) == p_number,
            func.cast(func.substr(models.Sample_Information.tissue_id, 2), Integer) == t_number
        ).first()

        if not sample_info:
            raise HTTPException(
                status_code=400,
                detail=f"No matching sample found for P{p_number} and T{t_number}. Please check the file: {file_name}"
            )

    # Process metadata files

    try:
        file = marker_file
        # Validate filename and extract P, T, R, S, B
        file_pattern = r"^P\d{5}-T\d{3}-R\d{3}-S\d{3}(-B\d)?(-\d+)?(-[A-Za-z_]{2,10})?\.marker$"
        match = re.match(file_pattern, file.filename)
        if not match:
            raise HTTPException(status_code=400,
                                detail="Invalid filename format for marker file. Expected format: P00001-T001-R001-S001(-B1)(-1).marker")
        print(1)
        file_parts = match.group(0)
        file_P = file_parts[0]
        file_T = file_parts[1]
        file_R = file_parts[2]
        file_S = file_parts[3]
        file_B = file_parts[4] if file_parts[4] else ''
        file_number = file_B.lstrip('-B') if file_B else ''

        # Validate sample number in the file name
        validate_sample_number(file.filename)
        print(2)
        # Read marker file content
        file.file.seek(0)
        lines = [line.decode('utf-8').strip() for line in file.file.readlines()]

        try:
            marker_df = read_marker_lines(lines)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading marker file '{file.filename}': {str(e)}")
        print(3)
        # Check required columns
        required_columns = {'name', 'x', 'y', 'z'}
        if not required_columns.issubset(marker_df.columns):
            raise HTTPException(status_code=400,
                                detail=f"Marker file '{file.filename}' is missing required columns: {required_columns}")
        print(4)
        # Validate 'name' column entries
        for name_entry in marker_df['name']:
            # Add a type check for name_entry
            if not isinstance(name_entry, str):
                print(name_entry)
                raise HTTPException(
                    status_code=400,
                    detail=f"'{file.filename}' File name and its 'name' column do not match."
                )
            name_pattern = r"^(P\d{5})_(T\d{3})_(R\d{3})_(S\d{3})(_B\d)??(_\d+)?(_[A-Za-z_]{2,10})?_C\d+"
            name_match = re.match(name_pattern, name_entry)
            if not name_match:
                raise HTTPException(status_code=400,
                                    detail=f"Invalid 'name' entry in marker file '{file.filename}': {name_entry}")
            name_parts = name_match.group(0)
            name_P = name_parts[0]
            name_T = name_parts[1]
            name_R = name_parts[2]
            name_S = name_parts[3]
            name_B = name_parts[4] if name_parts[4] else ''
            name_number = name_B.lstrip('_B') if name_B else ''

            # Compare file_P with name_P, etc.
            if (
                    file_P != name_P or file_T != name_T or file_R != name_R or file_S != name_S or file_number != name_number):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{file.filename}' File name and its 'name' column do not match. Please check."
                )
        print(5)
        # Step 1: Check for duplicate 'C' identifiers
        if marker_df['name'].duplicated().any():
            duplicate_names = marker_df[marker_df['name'].duplicated()]['name'].unique()
            raise HTTPException(
                status_code=400,
                # detail=f"Duplicate 'C' identifiers found in marker file '{file.filename}': {', '.join(duplicate_names)}"
                detail="Duplicate C numbers found in 'name' column. Please check."
            )

        # Step 2: Validate 'name' column format
        invalid_names = marker_df[
            ~marker_df['name'].str.match(r'^P\d{5}_T\d{3}_R\d{3}_S\d{3}(_B\d)?(_\d+)?(_[A-Za-z_]{2,10})?_C\d{5}$',
                                         na=False)]
        if not invalid_names.empty:
            raise HTTPException(
                status_code=400,
                # detail=f"Invalid 'name' entries in marker file '{file.filename}': {', '.join(invalid_names['name'].unique())}"
                detail="Invalid C number in the 'name' column. Please check."
            )
        if imaging_id == '--':
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}"
        else:
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}-{imaging_id}"
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, file.filename)
        # Save file
        file.file.seek(0)
        with open(file_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    except HTTPException as e:
        # Raise exception with uploaded_files
        raise HTTPException(status_code=400, detail={"error": e.detail, "uploaded_files": uploaded_files})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "uploaded_files": uploaded_files})
    details = json.dumps(file_path)
    crud.create_user_log(db, int(user_id),
                         f"Upload imaging_match_table of: {sample_preparation_id}",
                         details=details)
    return JSONResponse(content={"message": "File uploaded successfully", "uploaded_files": uploaded_files})


@app.post("/api/upload_imaging_match_table/{sample_preparation_id}/{imaging_id}")
async def upload_imaging_match_table(
        matchtable_file: UploadFile = File,
        sample_preparation_id: str = '',
        imaging_id: str = '',Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)  # 注入数据库会话
):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    uploaded_files = []

    # Function to validate sample numbers against the database
    def validate_sample_number(file_name: str):
        # 提取文件名中的 P 和 T 编号
        file_pattern = r"^P(\d{5})-T(\d{3})-R\d{3}-S\d{3}(-B\d)?(-\d+)?"
        match = re.match(file_pattern, file_name)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filename format for sample number check: {file_name}"
            )
        p_number, t_number = int(match.group(1)), int(match.group(2))
        print(p_number, t_number)

        # 查询数据库，验证 P 和 T 编号是否存在
        sample_info = db.query(models.Sample_Information).filter(
            func.cast(func.substr(models.Sample_Information.patient_number, 2), Integer) == p_number,
            func.cast(func.substr(models.Sample_Information.tissue_id, 2), Integer) == t_number
        ).first()

        if not sample_info:
            raise HTTPException(
                status_code=400,
                detail=f"No matching sample found for P{p_number} and T{t_number}. Please check the file: {file_name}"
            )

    # Process metadata files

    try:
        file = matchtable_file
        # Validate filename format
        print('filename', file.filename)
        if not re.match(r"^P\d{5}-T\d{3}-R\d{3}-S\d{3}(-B\d)?(-\d+)?(-[A-Za-z_]{2,10})?-matched\.csv$", file.filename):
            raise HTTPException(status_code=400,
                                detail="Invalid filename format for match table file. Expected format: P00001-T001-R001-S001(-B1)(-1)(-NAME)-matched.csv")

        # Validate sample number in the file name
        print('1')
        validate_sample_number(file.filename)
        print('2')
        if imaging_id == '--':
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}"
        else:
            folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}-{imaging_id}"
        os.makedirs(folder, exist_ok=True)
        # Check if file already exists
        file_path = os.path.join(folder, file.filename)
        # if os.path.exists(file_path):
        #     raise HTTPException(status_code=400, detail=f"'{file.filename}' already exists. Please check.")

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    except HTTPException as e:
        # Raise exception with uploaded_files
        raise HTTPException(status_code=400, detail={"error": e.detail, "uploaded_files": uploaded_files})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "uploaded_files": uploaded_files})
    details = json.dumps(file_path)
    crud.create_user_log(db, int(user_id),
                         f"Upload imaging_match_table of: {sample_preparation_id}",
                         details=details)
    return JSONResponse(content={"message": "File uploaded successfully", "uploaded_files": uploaded_files})


@app.get("/api/sample_preparation", response_model=List[SamplePreparationSchema])
def get_all_samples(db: Session = Depends(get_db)):
    return db.query(SamplePreparation).order_by(SamplePreparation.created_at.desc()).all()


@app.get("/api/sample_preparation/{id}", response_model=SamplePreparationSchema)
def get_sample(id: int, db: Session = Depends(get_db)):
    db_sample = db.query(SamplePreparation).filter(SamplePreparation.id == id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return db_sample


@app.post("/api/sample_preparation", response_model=SamplePreparationSchema)
def create_sample(sample: SamplePreparationSchema, Authorize: AuthJWT = Depends(),db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    # 检查是否已存在完全相同的记录
    existing_sample = db.query(SamplePreparation).filter(
        SamplePreparation.sampleId == sample.sampleId,
        SamplePreparation.tissueId == sample.tissueId,
        SamplePreparation.rollId == sample.rollId,
        SamplePreparation.sliceId == sample.sliceId,
        SamplePreparation.blockId == sample.blockId
    ).first()

    if existing_sample:
        raise HTTPException(
            status_code=400,
            detail="A record with the same PTRSB already exists."
        )

    # 检查是否存在相同 SampleId、TissueId、RollId、SliceId 且 BlockId 为 '--' 的记录
    conflict_sample = db.query(SamplePreparation).filter(
        SamplePreparation.sampleId == sample.sampleId,
        SamplePreparation.tissueId == sample.tissueId,
        SamplePreparation.rollId == sample.rollId,
        SamplePreparation.sliceId == sample.sliceId,
        SamplePreparation.blockId == '--'
    ).first()

    if conflict_sample:
        raise HTTPException(
            status_code=400,
            detail="A record with the same PTRS but BlockID='--' already exists. Please check your data."
        )

    # 插入新记录
    db_sample = SamplePreparation(
        sampleId=sample.sampleId,
        tissueId=sample.tissueId,
        rollId=sample.rollId,
        sliceId=sample.sliceId,
        blockId=sample.blockId,
        injected_num = sample.injected_num,
        dyes=sample.dyes,
        needles=sample.needles,
        status=sample.status,
        perfusion_user=sample.perfusion_user,
        perfusion_date=sample.perfusion_date,
        dye_name=sample.dye_name
    )
    db.add(db_sample)
    db.commit()
    db.refresh(db_sample)
    # details = json.dumps(db_sample.sampleId)
    crud.create_user_log(db, int(user_id),
                         f"create an new sample of: {id}")
    return db_sample


@app.put("/api/sample_preparation/{id}", response_model=SamplePreparationSchema)
def update_sample(id: int, sample: SamplePreparationSchema, Authorize: AuthJWT = Depends(),db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    db_sample = db.query(SamplePreparation).filter(SamplePreparation.id == id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    db_sample.sampleId = sample.sampleId
    db_sample.tissueId = sample.tissueId
    db_sample.rollId = sample.rollId
    db_sample.sliceId = sample.sliceId
    db_sample.blockId = sample.blockId
    db_sample.injected_num = sample.injected_num
    db_sample.dyes = sample.dyes
    db_sample.needles = sample.needles
    db_sample.status = sample.status
    db_sample.perfusion_user = sample.perfusion_user
    db_sample.perfusion_date = sample.perfusion_date
    db_sample.comment = sample.comment


    db.commit()
    db.refresh(db_sample)
    details = SamplePreparationSchema.from_orm(db_sample).json()
    crud.create_user_log(db, int(user_id),
                         f"modify the sample preparation of: {id}",
                         details=details)
    return db_sample


@app.delete("/api/sample_preparation/{id}")
def delete_sample(id: int, Authorize: AuthJWT = Depends(),db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    db_sample = db.query(SamplePreparation).filter(SamplePreparation.id == id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    # 删除SamplePreparation会自动删除关联的ImagingRecord（由于cascade和ondelete设置）
    db.delete(db_sample)
    db.commit()
    details = SamplePreparationSchema.from_orm(db_sample).json()
    crud.create_user_log(db, int(user_id),
                         f"delete the sample preparation of: {id}",
                         details=details)
    return {"message": f"SamplePreparation {id} deleted successfully."}

######################
# ImagingRecord CRUD #
######################

@app.get("/api/imaging_records", response_model=List[ImagingRecordSchema])
def get_all_imaging_records(db: Session = Depends(get_db)):
    return db.query(ImagingRecord).all()


@app.get("/api/imaging_records/{sample_preparation_id}/{imaging_id}", response_model=ImagingRecordSchema)
def get_imaging_record(sample_preparation_id: int, imaging_id: str, db: Session = Depends(get_db)):
    record = (
        db.query(ImagingRecord)
        .filter(
            ImagingRecord.sample_preparation_id == sample_preparation_id,
            ImagingRecord.imaging_id == imaging_id,
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="ImagingRecord not found")
    return record

def delete_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {file_path}. Details: {e}")

@app.get("/api/download_imaging_records_files/{sample_preparation_id}/{imaging_id}")
def download_imaging_records_files(sample_preparation_id: str, imaging_id: str, background_tasks: BackgroundTasks):
    try:
        BASE_DIR = "/mnt/nfs/hndb/SamplePreparation"
        # 构建文件夹路径
        sample_folder = os.path.join(BASE_DIR, sample_preparation_id)
        if imaging_id == '--':
            imaging_folder = os.path.join(sample_folder, f"{sample_preparation_id}")
        else:
            imaging_folder = os.path.join(sample_folder, f"{sample_preparation_id}-{imaging_id}")

        # 校验文件夹是否存在
        if not os.path.exists(sample_folder) or not os.path.exists(imaging_folder):
            raise HTTPException(status_code=404, detail="Sample or imaging folder not found.")

        # 定义需要的文件扩展名
        required_extensions = [".xlsx", ".xml", "_MIP.tif", ".marker", ".apo"]
        map_extensions = [".png", ".jpg", ".jpeg"]

        # 收集需要打包的文件
        files_to_zip = []

        # 搜索 imaging_folder 中的文件
        for file_name in os.listdir(imaging_folder):
            for ext in required_extensions:
                if file_name.endswith(ext):
                    files_to_zip.append(os.path.join(imaging_folder, file_name))
                    break  # 防止重复匹配相同文件

        # 搜索 sample_folder 中的 _map 图片文件
        for file_name in os.listdir(sample_folder):
            if any(file_name.endswith(ext) for ext in map_extensions) and "_map" in file_name:
                files_to_zip.append(os.path.join(sample_folder, file_name))

        # 如果没有文件需要打包，返回错误
        if not files_to_zip:
            raise HTTPException(status_code=404, detail="No matching files found to zip.")

        # 创建临时压缩包路径
        if imaging_id == "--":
            zip_filename = f"{sample_preparation_id}_files.zip"
        else:
            zip_filename = f"{sample_preparation_id}_{imaging_id}_files.zip"

        zip_path = os.path.join(sample_folder, zip_filename)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                # 添加文件到压缩包，保持目录结构平坦（只保留文件名）
                zipf.write(file_path, arcname=os.path.basename(file_path))

        # 使用 BackgroundTask 延迟删除文件
        background_tasks.add_task(delete_file, zip_path)

        # 返回压缩包文件
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_filename,
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/imaging_records", response_model=ImagingRecordSchema)
def create_imaging_record(record: ImagingRecordSchema, Authorize: AuthJWT = Depends(),db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    # 验证父表 ID 是否存在
    db_sample = db.query(SamplePreparation).filter(SamplePreparation.id == record.sample_preparation_id).first()
    if not db_sample:
        raise HTTPException(status_code=400, detail="Invalid sample_preparation_id")

    # 检查组合唯一性
    existing_record = (
        db.query(ImagingRecord)
        .filter(
            ImagingRecord.sample_preparation_id == record.sample_preparation_id,
            ImagingRecord.imaging_id == record.imaging_id,
        )
        .first()
    )
    if existing_record:
        raise HTTPException(status_code=400,
                            detail="Duplicate ImagingRecord for this sample_preparation_id and imaging_id")

    conflict_record = (
        db.query(ImagingRecord)
        .filter(
            ImagingRecord.sample_preparation_id == record.sample_preparation_id,
            ImagingRecord.imaging_id == '--',
        )
        .first()
    )

    # 检查是否存在 imaging_id 为数字标号的记录
    numeric_record_exists = (
        db.query(ImagingRecord)
        .filter(
            ImagingRecord.sample_preparation_id == record.sample_preparation_id,
            ImagingRecord.imaging_id != '--',
        )
        .first()
    )

    # 如果存在冲突记录或者存在数字标号记录且上传的是 '--'
    if conflict_record or (numeric_record_exists and record.imaging_id == '--'):
        raise HTTPException(
            status_code=400,
            detail="Invalid ImagingRecord: imaging_id cannot be '--' when numeric imaging_id records exist."
        )


    # 创建记录
    new_record = ImagingRecord(
        imaging_id=record.imaging_id,
        sample_preparation_id=record.sample_preparation_id,
        producer=record.producer,
        status=record.status,
        Dyes=1,
        Z_Size=0.0,
        Y_Size=0.0,
        X_Size=0.0,
        File_Size_GB=0.0
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    # details = json.dumps(ImagingRecordSchema.from_orm(new_record).dict())
    crud.create_user_log(db, int(user_id), f"create an new imaging_record of: {record.sample_preparation_id}-{record.imaging_id}")
    return new_record


@app.put("/api/imaging_records/{sample_preparation_id}/{imaging_id}", response_model=ImagingRecordSchema)
def update_imaging_record(
        sample_preparation_id: int,
        imaging_id: str,
        record: ImagingRecordSchema,Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db),
):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    db_record = (
        db.query(ImagingRecord)
        .filter(
            ImagingRecord.sample_preparation_id == sample_preparation_id,
            ImagingRecord.imaging_id == imaging_id,
        )
        .first()
    )
    if not db_record:
        raise HTTPException(status_code=404, detail="ImagingRecord not found")

    # 更新记录
    db_record.producer = record.producer
    db_record.status = record.status

    db.commit()
    db.refresh(db_record)
    # details = json.dumps(ImagingRecordSchema.from_orm(db_record).dict())
    crud.create_user_log(db, int(user_id), f"modify the imaging_records of: {sample_preparation_id}-{imaging_id}")
    return db_record


@app.delete("/api/imaging_records/{sample_preparation_id}/{imaging_id}")
def delete_imaging_record(sample_preparation_id: int, imaging_id: str, Authorize: AuthJWT = Depends(),db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()

    db_record = (
        db.query(ImagingRecord)
        .filter(
            ImagingRecord.sample_preparation_id == sample_preparation_id,
            ImagingRecord.imaging_id == imaging_id,
        )
        .first()
    )
    if not db_record:
        raise HTTPException(status_code=404, detail="ImagingRecord not found")

    db.delete(db_record)
    db.commit()
    # details = json.dumps(ImagingRecordSchema.from_orm(db_record).dict())
    crud.create_user_log(db, int(user_id), f"Delete imaging_records with id: {sample_preparation_id}-{imaging_id}")
    return {
        "message": f"ImagingRecord with imaging_id '{imaging_id}' and sample_preparation_id '{sample_preparation_id}' deleted successfully."}


@app.post("/api/upload_imaging_map")
async def upload_imaging_map(imaging_map_file: UploadFile = File,Authorize: AuthJWT = Depends(),db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    responses = []
    file_name = imaging_map_file.filename
    if "_map" in file_name:
        folder_name = file_name.split("_map")[0]  # 提取-map之前的部分
    else:
        return JSONResponse(content={"message": "Invalid file name format", "file": file_name}, status_code=400)
    # 构建保存路径
    base_upload_dir = f"/mnt/nfs/hndb/SamplePreparation/{folder_name}"
    os.makedirs(base_upload_dir, exist_ok=True)  # 确保文件夹存在

    # 保存文件
    file_location = os.path.join(base_upload_dir, file_name)
    with open(file_location, "wb+") as file_object:
        file_object.write(await imaging_map_file.read())

    responses.append(file_location)
    details = json.dumps(file_location)
    crud.create_user_log(db, int(user_id), f"Upload imaging_map of: {folder_name}",
                         details=details)
    return JSONResponse(content={"message": "Upload successful!", "files": responses})


@app.get("/api/get_imaging_map/{sample_preparation_id}")
def get_imaging_map(sample_preparation_id: str):
    base_path = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}"

    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    # 检查是否存在 -map 文件
    for ext in ["png", "jpeg", "jpg"]:
        file_path = os.path.join(base_path, f"{sample_preparation_id}_map.{ext}")
        if os.path.exists(file_path):
            # 直接返回文件
            return FileResponse(file_path, media_type=f"image/{ext}", filename=os.path.basename(file_path))

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/get_imaging_mip/{sample_preparation_id}/{imaging_id}")
def get_imaging_mip(sample_preparation_id: str, imaging_id: str):
    base_path = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}"

    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    if imaging_id.isdigit():
        imaging_id_part = f"-{imaging_id}"
    elif imaging_id == "--":
        imaging_id_part = ""
    else:
        raise HTTPException(status_code=400, detail="Invalid imaging_id")
    base_path = f"{base_path}/{sample_preparation_id}{imaging_id_part}"
    # 检查是否存在 -map 文件
    for ext in ["tif"]:
        file_path = os.path.join(base_path, f"{sample_preparation_id}{imaging_id_part}_MIP.{ext}")
        if file_path.endswith('.tif'):
            jpg_file_path = file_path.replace('.tif', '.jpg')
            if not os.path.exists(jpg_file_path):
                with Image.open(file_path) as img:
                    img.convert("RGB").save(jpg_file_path, "JPEG")
            if os.path.exists(jpg_file_path):
                # 直接返回文件
                return FileResponse(jpg_file_path, media_type=f"image/{ext}", filename=os.path.basename(file_path))

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/get_injection_file/{sample_preparation_id}")
def get_injection_file(sample_preparation_id: str):
    base_path = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}"

    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    # 检查是否存在 -map 文件
    for ext in ["csv"]:
        file_path = os.path.join(base_path, f"{sample_preparation_id}.{ext}")
        if os.path.exists(file_path):
            # 直接返回文件
            return FileResponse(file_path, media_type=f"text/{ext}", filename=os.path.basename(file_path),
                                headers={"Content-Disposition": f"attachment; filename={sample_preparation_id}.{ext}"})

    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/api/download_injection_records")
async def download_sample_preparation(db: Session = Depends(get_db)):
    try:
        # 执行查询
        query = text("SELECT * FROM sample_preparation")
        result = db.execute(query)
        rows = result.fetchall()

        # 获取列名
        column_names = result.keys()

        # 写入到内存中的 CSV 文件
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(column_names)  # 写入表头
        for row in rows:
            writer.writerow(row)

        # 重置内存中的文件指针
        output.seek(0)

        # 以 StreamingResponse 形式返回 CSV
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=injection_records.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/api/download_imaging_records")
async def download_sample_preparation(db: Session = Depends(get_db)):
    try:
        # 执行查询
        query = text('SELECT * FROM imaging_records')
        result = db.execute(query)
        rows = result.fetchall()

        # 获取列名
        column_names = result.keys()

        # 写入到内存中的 CSV 文件
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(column_names)  # 写入表头
        for row in rows:
            writer.writerow(row)

        # 重置内存中的文件指针
        output.seek(0)

        # 以 StreamingResponse 形式返回 CSV
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=imaging_records.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/upload_imaging_data/{sample_preparation_id}/{imaging_id}")
async def upload_imaging_data(imaging_data_file: UploadFile = File, sample_preparation_id: str = '',Authorize: AuthJWT = Depends(),
                              imaging_id: str = '',db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    responses = []
    file_name = imaging_data_file.filename
    # 构建保存路径
    if imaging_id == '--':
        base_upload_dir = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}"
    else:
        base_upload_dir = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}-{imaging_id}"
    os.makedirs(base_upload_dir, exist_ok=True)

    # 保存文件
    file_location = os.path.join(base_upload_dir, file_name)
    with open(file_location, "wb+") as file_object:
        file_object.write(await imaging_data_file.read())

    responses.append(file_location)
    details = json.dumps(file_location)
    crud.create_user_log(db, int(user_id), f"Upload imaging_data of: {sample_preparation_id}",
                         details=details)

    return JSONResponse(content={"message": "Upload successful!", "files": responses})


@app.post("/api/upload_bright_field_data/{sample_preparation_id}")
async def upload_bright_field_data(
        sample_preparation_id: str,
        Authorize: AuthJWT = Depends(),
        bright_field_data_files: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    base_upload_dir = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}"
    os.makedirs(base_upload_dir, exist_ok=True)  # 确保文件夹存在

    saved_files = []
    for bf_file in bright_field_data_files:
        file_name = bf_file.filename
        print(file_name)
        file_path = os.path.join(base_upload_dir, file_name)

        # 保存文件
        with open(file_path, "wb") as file_object:
            file_object.write(await bf_file.read())

        saved_files.append(file_path)
    details = json.dumps(saved_files)
    crud.create_user_log(db, int(user_id), f"Upload bright_field_data of: {sample_preparation_id}",
                         details=details)
    return JSONResponse(content={
        "message": "Upload successful!",
        "uploaded_files": saved_files
    })

REQUIRED_COLUMNS_NEW = [
    "Id",
    "sample_preparation_date",
    "sample_preparation_time",
    "sample_preparation_staff",
    "slice_thickness",
    "fresh_perfusion",
    "Status",
    "dye_name",
    "dye_concentration(%)",
    "immunohistochemistry",
    "ihc_category",
    "primaryAntibody_concentration",
    "secondAntibody_band",
    "DAPI_concentration",
    "perfusion_date",
    "perfusion_time",
    "AddingTime",
    "Depth",
    "current_intensity",
    "perfusion_time_on",
    "perfusion_time_off",
    "experiment_temperature",
    "experiment_humidity",
    "perfusion_user",
    "X",
    "Y",
    "Z",
    "AddingX",
    "AddingY",
    "AddingZ",
    # "FLresult",
    # "Needle_name"
]


@app.get("/api/get_injection_ids/{sample_preparation_id}", response_model=List[str])
async def get_injection_ids(sample_preparation_id: str):
    """
    获取指定样本灌注表中的 Id 列内容
    :param sample_preparation_id: 当前样本的 ID，例如 P00079-T001-R002-S026
    :return: 返回 Id 列的内容列表
    """
    # 替换 sample_id 中的 `-` 为 `_`，以匹配文件命名规则
    file_name = f"{sample_preparation_id}.csv"
    folder = os.path.join("/mnt/nfs/hndb/SamplePreparation", sample_preparation_id)
    file_path = os.path.join(folder, file_name)
    print(file_path)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Injection table for {sample_preparation_id} not found.")

    try:
        # 读取 CSV 文件并提取 Id 列
        id_list = []
        with open(file_path, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if "Id" not in reader.fieldnames:
                raise HTTPException(status_code=400, detail="CSV file does not contain 'Id' column.")
            for row in reader:
                id_list.append(row["Id"])
        return id_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process injection table: {str(e)}")


@app.post("/api/insert_injection_file_to_db/{sample_preparation_id}")
async def insert_injection_file_to_db(sample_preparation_id: str, Authorize: AuthJWT = Depends(),db: Session = Depends(get_db)):
    Authorize.jwt_required()
    user_id = Authorize.get_jwt_subject()
    file_path = "/mnt/nfs/hndb/SamplePreparation"
    # 构建文件路径
    file_path = os.path.join(file_path, sample_preparation_id, f"{sample_preparation_id}.csv")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at {file_path}")

    try:
        # 读取 CSV 文件内容并将其转换为 DataFrame
        df = pd.read_csv(file_path, encoding='utf-8')

        # 检查 CSV 文件中是否有 ID 列
        if 'Id' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain an ID column.")

        # 校验 CSV 文件数据格式
        file_name_without_ext = sample_preparation_id
        pattern = re.match(r"^(P\d{5})-(T\d{3})-(R\d{3})-(S\d{3})(?:-(B\d))?$", file_name_without_ext)
        if not pattern:
            raise HTTPException(status_code=400, detail="Invalid sample_preparation_id format.")

        # 提取 P, T, R, S, (B) 部分
        p_part, t_part, r_part, s_part, b_part = pattern.groups()
        prefix = f"{p_part}_{t_part}_{r_part}_{s_part}"
        if b_part:
            prefix += f"_{b_part}"
        prefix_pattern = re.compile(rf"^{re.escape(prefix)}_C\d+$")

        # 检查 ID 列中的所有值是否符合文件名中的格式
        if not df['Id'].apply(lambda x: bool(prefix_pattern.match(str(x)))).all():
            raise HTTPException(status_code=400, detail="File name does not match its ID column.")

        # 检查是否有重复的 C 编号
        c_numbers = df['Id'].apply(lambda x: re.search(r"C\d{5}$", str(x)).group())
        if c_numbers.duplicated().any():
            raise HTTPException(status_code=400, detail="Duplicate C numbers found in ID column.")

        # 检查所有必需列是否存在
        REQUIRED_COLUMNS = ['Id', 'dye_name', 'sample_preparation_date', 'perfusion_date']  # 示例必需列
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")

        # 检查日期列格式
        date_columns = ['sample_preparation_date', 'perfusion_date']
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='raise', infer_datetime_format=True)
            except ValueError as e:
                raise HTTPException(status_code=400,
                                    detail=f"Unable to convert date format in column {date_col}: {str(e)}")

        # 修改：处理 ihc_category 列
        cutoff_date = pd.to_datetime('2024-10-29')

        def process_ihc_category(row):
            if row['perfusion_date'] <= cutoff_date:
                # 现有逻辑
                if row['dye_name'] == 'Lucifer Yellow':
                    return 'Lucifer Yellow'
                else:
                    return '-'
            else:
                # 保留原始值，不进行处理
                return row['ihc_category']

        df['ihc_category'] = df.apply(process_ihc_category, axis=1)

        df['sample_preparation_date'] = df['sample_preparation_date'].dt.strftime('%Y-%m-%d')
        df['perfusion_date'] = df['perfusion_date'].dt.strftime('%Y-%m-%d')

        # 添加一个新列 file_name 并将所有行的值设置为当前文件名
        df['file_name'] = f'{sample_preparation_id}.csv'

        df = df.replace({np.nan: '--'})

        # 尝试将 DataFrame 插入数据库
        table = Table('injection_table_20241028', MetaData(), autoload_with=db.bind)
        try:
            for _, row in df.iterrows():
                # print(row)
                stmt = insert(table).values(row.to_dict())
                db.execute(stmt)

            db.commit()

        except SQLAlchemyError as db_error:
            print(f"Database insertion failed: {db_error}")
            raise HTTPException(status_code=400, detail=f"Database insertion failed: {str(db_error)}")


    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")
    details = json.dumps(file_path)
    crud.create_user_log(db, int(user_id), f"insert injection_file of: {sample_preparation_id} to db",
                         details=details)
    return {"message": "CSV uploaded and stored in the database successfully"}


@app.get("/api/check_sample_file_exists")
async def check_sample_file_exists(filename: str):
    folder = f"/mnt/nfs/hndb/SamplePreparation/{filename.split('.')[0]}"
    file_path = os.path.join(folder, filename)
    print(folder)
    if os.path.exists(file_path):
        return {"exists": True}
    return {"exists": False}


@app.get("/api/check_imaging_record_file_exists/{sample_preparation_id}/{imaging_id}")
async def check_imaging_record_file_exists(filename: str, sample_preparation_id: str, imaging_id: str):
    # 提取文件名中的基础部分（不包含扩展名）
    if imaging_id == '--':
        folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}"
    else:
        folder = f"/mnt/nfs/hndb/SamplePreparation/{sample_preparation_id}/{sample_preparation_id}-{imaging_id}"
    # 生成完整文件路径
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        return {"exists": True}
    return {"exists": False}


### LLMs 部分

# 配置 Redis
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
REDIS_EXPIRATION_TIME = 1800  # 上下文过期时间，单位：秒（30分钟）


class Question(BaseModel):
    session_id: str
    question: str


@app.post("/api/agent")
async def agent_endpoint(question: Question):
    try:
        # 验证数据是否有效
        if not question.session_id or not question.question:
            raise HTTPException(status_code=422, detail="Missing session_id or question")

        # 调用多轮对话处理函数
        answer = await process_user_question(question.session_id, question.question)
        return {"answer": answer}
    except Exception as e:
        # 捕获所有异常并记录日志
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def process_user_question(session_id: str, question: str) -> str:
    """
    Processes the user's question, supports multi-turn conversations.

    :param session_id: Unique identifier for the session, used for context storage
    :param question: User's question
    :return: Assistant's answer
    """
    # Retrieve context from Redis
    context = get_context_from_redis(session_id)

    # Append user's question to context
    context.append({"role": "user", "content": question})

    # Determine question category
    category = await determine_question_category(question)
    print(f"Category: {category}")

    # Map category to handler functions
    category_handlers = {
        1: handle_database_status,
        2: handle_data_production,
        3: handle_simple_field_query,
        4: handle_detailed_field_report,
        5: handle_professional_question,
    }

    handler = category_handlers.get(category, handle_professional_question)
    answer = await handler(question, context)

    # Append assistant's answer to context
    context.append({"role": "assistant", "content": answer})
    save_context_to_redis(session_id, context)

    return answer


def get_context_from_redis(session_id: str) -> list:
    """
    从 Redis 中获取上下文。
    
    :param session_id: 会话 ID
    :return: 上下文列表
    """
    context = redis_client.get(session_id)
    if context:
        return json.loads(context)
    # 如果上下文不存在，初始化默认值
    return [{"role": "system", "content": "You are a helpful assistant specialized in neuroscience databases."}]


def save_context_to_redis(session_id: str, context: list):
    """
    将上下文保存到 Redis, 并设置过期时间。
    
    :param session_id: 会话 ID
    :param context: 上下文列表
    """
    redis_client.set(session_id, json.dumps(context), ex=REDIS_EXPIRATION_TIME)


# def determine_question_category(question: str) -> int:
#     """
#     Determines the category of the question based on predefined keywords.
#     Categories:
#     1 - Database status
#     2 - Data production situation
#     3 - Simple field queries
#     4 - Detailed field reports
#     5 - Professional questions
#     """
#     normalized_question = question.lower()

#     category_keywords = {
#         1: ["data status", "database status", "database summary", "data overview", "database info", "database information", "overview of the database"],
#         2: ["data production", "daily production", "production report"],
#         3: ["number of", "count of", "how many", "quantity of", "total number"],
#         4: ["distribution", "overall situation", "list all", "summary of", "detailed report"],
#     }

#     for category, keywords in category_keywords.items():
#         if any(keyword in normalized_question for keyword in keywords):
#             return category
#     # Default to category 5 if no keywords match
#     return 5

async def determine_question_category(question: str) -> int:
    """
    Determines the category of the question using an LLM.
    Categories:
    1 - Database status
    2 - Data production situation
    3 - Simple field queries
    4 - Detailed field reports
    5 - Professional questions
    """
    # Build a prompt for the LLM
    prompt = (
        "Please classify the following question into one of the following categories by providing the category number only:\n\n"
        "Categories:\n"
        "1 - Database status queries (e.g., 'What is the current status of the database?')\n"
        "2 - Data production situation (e.g., 'What data was produced today?')\n"
        "3 - Simple field queries requiring a numerical answer (e.g., 'How many cells are in region 'MFG'?')\n"
        "4 - Detailed field reports requiring a summary or distribution (e.g., 'Provide a report on the age distribution of samples.')\n"
        "5 - Professional or general questions (e.g., 'Explain the role of neurons in the brain.')\n\n"
        f"Question: \"{question}\"\n\n"
        "Provide only the category number (1-5)."
    )

    # Call the LLM with the prompt
    response = await call_large_language_model(prompt)

    # print(response)

    # Extract the category number from the response
    try:
        category = int(response.strip())
        if category in [1, 2, 3, 4, 5]:
            return category
        else:
            # If the response is not a valid category number, default to category 5
            return 5
    except ValueError:
        # If the response cannot be converted to an integer, default to category 5
        return 5


async def handle_database_status(question: str, context: list) -> str:
    data_status = await get_data_status()
    return format_data_status(data_status)


async def handle_data_production(question: str, context: list) -> str:
    date = extract_date_from_question(question)
    if not date:
        return "Please provide a specific date for the production report."
    perfusion_count = await get_perfusion_count_by_date(date)
    imaging_count = await get_imaging_count_by_date(date)
    return (f"On {date.strftime('%Y-%m-%d')}, there were {perfusion_count} perfusion records "
            f"and {imaging_count} imaging records.")


async def handle_simple_field_query(question: str, context: list) -> str:
    field, value = extract_field_and_value(question)
    if not field or not value:
        return "Could not understand the field or value in your question. Please specify clearly."
    count = await get_count_by_field_value(field, value)
    return f"The number of records where '{field}' is '{value}' is {count}."


async def handle_detailed_field_report(question: str, context: list) -> str:
    field = extract_field(question)
    if not field:
        return "Could not understand the field you are interested in. Please specify clearly."
    data_list = await get_field_data(field)
    report = await generate_field_report(data_list, field)
    return report


async def handle_professional_question(question: str, context: list) -> str:
    return await call_large_language_model_with_context(question, context)


def extract_date_from_question(question: str):
    cal = parsedatetime.Calendar()
    time_struct, parse_status = cal.parse(question)
    if parse_status == 1:
        return datetime(*time_struct[:6]).date()
    return None


nlp = spacy.load('en_core_web_sm')


def extract_field_and_value(question: str):
    doc = nlp(question.lower())
    field_synonyms = {
        "age": ["age", "ages", "old"],
        "gender": ["gender", "sex"],
        "brain region": ["brain region", "region", "brain area", "area"],
        "sample source": ["sample source", "source", "origin"],
    }
    field = None
    value = None
    for token in doc:
        for key, synonyms in field_synonyms.items():
            if token.text in synonyms:
                field = key
                break
        if field:
            break
    if not field:
        return None, None
    # 寻找值，假设在字段后面
    token_index = token.i
    for token in doc[token_index + 1:]:
        if token.pos_ in ["NOUN", "PROPN", "NUM", "ADJ"]:
            value = token.text
            break
    return field, value


def extract_field(question: str):
    doc = nlp(question.lower())
    field_synonyms = {
        "age": ["age", "ages", "age distribution"],
        "gender": ["gender", "sex", "genders"],
        "brain region": ["brain region", "region", "regions", "brain area", "areas"],
        "sample source": ["sample source", "source", "sources", "origin", "origins"],
    }
    field = None
    for token in doc:
        for key, synonyms in field_synonyms.items():
            if token.text in synonyms:
                field = key
                break
        if field:
            break
    return field


async def get_data_status():
    db = SessionLocal()
    try:
        total_samples = crud.get_total_samples(db)
        valid_samples = crud.get_valid_samples(db)
        total_cells = crud.get_total_count(db)
        total_regions = crud.get_total_regions(db)
        data_status = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "cells": total_cells,
            "regions": total_regions
        }
    finally:
        db.close()
    return data_status


def format_data_status(data_status: dict) -> str:
    return (
        f"The database contains {data_status['total_samples']} samples, "
        f"{data_status['valid_samples']} valid samples, "
        f"{data_status['cells']} cells, "
        f"spanning {data_status['regions']} brain regions."
    )


async def get_perfusion_count_by_date(date: datetime.date) -> int:
    db = SessionLocal()
    try:
        count = db.query(models.Injection_Table).filter(
            func.date(models.Injection_Table.perfusion_date) == date
        ).count()
        return count
    except Exception as e:
        print(f"Error querying perfusion data: {e}")
        return 0
    finally:
        db.close()


async def get_imaging_count_by_date(date: datetime.date) -> int:
    db = SessionLocal()
    try:
        count = db.query(models.Imaging_Information).filter(
            func.date(models.Imaging_Information.shooting_date) == date
        ).count()
        return count
    except Exception as e:
        print(f"Error querying imaging data: {e}")
        return 0
    finally:
        db.close()


async def get_count_by_field_value(field: str, value: str) -> int:
    db = SessionLocal()
    try:
        model_field = get_model_field(field)
        if not model_field:
            return 0
        count = db.query(model_field).filter(model_field == value).count()
        return count
    except Exception as e:
        print(f"Error querying data: {e}")
        return 0
    finally:
        db.close()


def get_model_field(field_name: str):
    """
    Maps field names to model fields.
    """
    field_mapping = {
        "age": models.Sample_Information.patient_age,
        "gender": models.Sample_Information.gender,
        "brain region": models.HumanSingleCellTrackingTable.brain_region,
        "sample source": models.Sample_Information.sample_id,
        "dye": models.HumanSingleCellTrackingTable.dye_name,
        "immunohistochemistry": models.HumanSingleCellTrackingTable.immunohistochemistry,
    }
    return field_mapping.get(field_name.lower())


async def get_field_data(field: str):
    db = SessionLocal()
    try:
        model_field = get_model_field(field)
        if not model_field:
            return []
        data = db.query(model_field).all()
        return [item[0] for item in data]
    except Exception as e:
        print(f"Error querying data: {e}")
        return []
    finally:
        db.close()


# def generate_field_report(data_list: list, field: str) -> str:
#     data_counts = Counter(data_list)
#     data_summary = "\n".join([f"{item}: {count}" for item, count in data_counts.items()])
#     return f"Here is the distribution of {field}:\n{data_summary}"

async def generate_field_report(data_list: list, field: str) -> str:
    """
    Generates a detailed report for the given field using LLM.
    """
    # Prepare data summary
    data_counts = Counter(data_list)
    # Convert counts to a string format suitable for LLM input
    data_summary = "\n".join([f"{item}: {count}" for item, count in data_counts.items()])

    # Construct a prompt for the LLM
    # prompt = (
    #     f"As an expert data analyst, please provide a brief report on the distribution of '{field}' "
    #     f"in the dataset based on the following information:\n\n"
    #     f"{data_summary}\n\n"
    # )

    prompt = (
        f"Based on the following {field} information, please provide a brief report:\n\n"
        f"{data_summary}\n\n"
    )

    # Call the LLM to generate the report
    report = await call_large_language_model_with_context(prompt, [])
    return report


async def call_large_language_model(prompt: str) -> str:
    """
    Calls the LLM to get a response based on the prompt.
    """
    response = await openai.ChatCompletion.acreate(
        model='gpt-4o',
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=5,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


async def call_large_language_model_with_context(prompt: str, context: list) -> str:
    """
    Calls the LLM, passing the context for multi-turn conversation.

    :param prompt: The prompt or question to send to the LLM
    :param context: Conversation context
    :return: Assistant's response
    """
    # Build the messages list for the API call
    messages = context + [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use an appropriate model name
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# def is_database_status_query(question: str) -> bool:
#     database_status_keywords = [
#         "data status",
#         "database status",
#         "database summary",
#         "data overview",
#         "database info",
#         "database information",
#         "数据库状态",
#         "数据库概况",
#         "数据概览",
#         "数据库信息",
#         "数据库摘要",
#         "数据状态",
#         "数据摘要",
#         "数据库目前情况",
#         "数据库内容",
#         "数据库统计",
#     ]
#     normalized_question = question.lower()
#     for keyword in database_status_keywords:
#         if keyword.lower() in normalized_question:
#             return True
#     return False

# def is_daily_production_query(question: str) -> bool:
#     daily_production_keywords = [
#         "日数据生产情况",
#         "每日数据生产",
#         "今天的数据生产",
#         "某天的数据生产",
#         "数据生产情况",
#         "数据生产统计",
#         "数据生产报告",
#         "data production",
#         "daily production",
#         "production report",
#     ]
#     normalized_question = question.lower()
#     for keyword in daily_production_keywords:
#         if keyword.lower() in normalized_question:
#             return True
#     return False

# def extract_date_from_question(question: str):
#     cal = parsedatetime.Calendar()
#     time_struct, parse_status = cal.parse(question)
#     if parse_status == 1:
#         # 解析成功，转换为日期对象
#         return datetime(*time_struct[:6]).date()
#     else:
#         return None

# async def generate_daily_production_report(question: str) -> str:
#     # 从用户的问题中提取日期
#     date = extract_date_from_question(question)
#     if not date:
#         return "Sorry, I couldn't extract a date from your question. Please provide a specific date, e.g., 'November 14, 2024'."

#     # 查询数据库获取数据
#     perfusion_count = await get_perfusion_count_by_date(date)
#     imaging_count = await get_imaging_count_by_date(date)

#     # 生成英文报告
#     report = (
#         f"On {date.strftime('%Y-%m-%d')}, there were {perfusion_count} perfusion records "
#         f"and {imaging_count} imaging records."
#     )
#     return report

# async def get_perfusion_count_by_date(date: datetime.date) -> int:
#     db = SessionLocal()
#     try:
#         count = db.query(models.Injection_Table).filter(
#             func.date(models.Injection_Table.perfusion_date) == date
#         ).count()
#         return count
#     except Exception as e:
#         print(f"Error querying perfusion data: {e}")
#         return 0
#     finally:
#         db.close()

# async def get_imaging_count_by_date(date: datetime.date) -> int:
#     db = SessionLocal()
#     try:
#         count = db.query(models.Imaging_Information).filter(
#             func.date(models.Imaging_Information.shooting_date) == date
#         ).count()
#         return count
#     except Exception as e:
#         print(f"Error querying imaging data: {e}")
#         return 0
#     finally:
#         db.close()

# Sample Information Retrieval
# async def get_sample_info(query_info):
#     db = SessionLocal()
#     try:
#         data_field = query_info["field"]
#         raw_data = db.query(data_field).all()
#     except Exception as e:
#         print(f"Error querying data: {e}")
#         return []
#     finally:
#         db.close()

#     # Extract data if an extract function is provided
#     if query_info["extract_func"]:
#         processed_data = [query_info["extract_func"](item[0]) for item in raw_data]
#     else:
#         processed_data = [item[0] for item in raw_data]

#     return processed_data

# def extract_sample_source(sample_id: str) -> str:
#     # Split sample_id by '-' and join the first two parts
#     return '-'.join(sample_id.split('-')[:2])

# # Report Generation
# async def generate_sample_info_report(data_list: list, description: str) -> str:
#     # Prepare a summary or data to send to the LLM
#     # For example, count occurrences of each sample source

#     data_counts = Counter(data_list)
#     data_summary = "\n".join([f"{item}: {count}" for item, count in data_counts.items()])

#     # Construct a prompt for the LLM
#     prompt = (
#         f"Based on the following {description} information, please provide a brief report:\n\n"
#         f"{data_summary}\n\n"
#     )

#     # Call the LLM with the prompt
#     response = await call_large_language_model_with_context(prompt, [])

#     return response

# # Define a mapping of query keywords to database fields and report generation functions
# QUERY_MAPPING = {
#     "sample source": {
#         "field": models.Sample_Information.sample_id,
#         "extract_func": extract_sample_source,
#         "report_func": generate_sample_info_report,
#         "description": "sample source",
#     },
#     "sample age": {
#         "field": models.Sample_Information.patient_age,
#         "extract_func": None,
#         "report_func": generate_sample_info_report,
#         "description": "sample age",
#     },
#     "sample gender": {
#         "field": models.Sample_Information.gender,
#         "extract_func": None,
#         "report_func": generate_sample_info_report,
#         "description": "sample gender",
#     },
#     "brain region": {
#         "field": models.HumanSingleCellTrackingTable.brain_region,
#         "extract_func": None,
#         "report_func": generate_sample_info_report,
#         "description": "brain region",
#     },
#     "dye": {
#         "field": models.HumanSingleCellTrackingTable.dye_name,
#         "extract_func": None,
#         "report_func": generate_sample_info_report,
#         "description": "dye",
#     },
#     "immunohistochemistry": {
#         "field": models.HumanSingleCellTrackingTable.immunohistochemistry,
#         "extract_func": None,
#         "report_func": generate_sample_info_report,
#         "description": "immunohistochemistry",
#     },
# }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

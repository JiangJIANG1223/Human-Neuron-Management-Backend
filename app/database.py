# 定义了数据库连接和会话管理

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = 'mysql+pymysql://root:wlj990521@10.194.35.182/hndb'
SQLALCHEMY_DATABASE_URL = 'mysql+pymysql://root:braintell%40seu@localhost/human_neuron'
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

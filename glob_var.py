import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from kafka import KafkaProducer
import os,sys
from minio import Minio

class GlobVar(object):
    dict_data = []
    arcface = None
    check_run = False
    CUDA = 'cuda:2'
import os
from utils import *
import warnings
warnings.filterwarnings("ignore")
random_state = 42


current_path = os.path.dirname(os.path.realpath(__file__)) # 전처리 실행 함수의 현재 경로
save_dir = os.path.join(current_path, 'data') 
yaml_path = os.path.join(current_path, 'data_path.yaml')  # json 경로가 저장된 yaml 파일 경로

os.makedirs(save_dir, exist_ok = True) #실행 경로 하위에 'data' 폴더 생성

dtypes = ['train', 'val', 'test'] # 3개의 csv파일 생성 후 저장

for dtype in dtypes:
    save_preprocessed_csv(yaml_path, save_dir, dtype) # from utils.py
 



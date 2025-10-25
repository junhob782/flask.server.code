#프로젝트에 필요한 설정값 ( 중앙 제어판 )

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "lotbotsystem"
}

SLOT_ROIS = {
    1: ((445, 614), (533, 693)),
    2: ((576 ,617), (765, 691)),
    3: ((717, 617), (995, 692)),
}

MODEL_NAME = 'vit_base_patch16_224' 
NUM_CLASSES = 2

# config.py
CCTV_SOURCE = r"C:\Users\hanhw\capstonedesign\lotbot_server\videos\1.mp4"
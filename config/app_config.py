import dotenv
import os
import numpy as np
import pandas as pd

dotenv.load_dotenv()

# VAR_IMU_F = float(os.getenv("VAR_IMU_F"))
VAR_IMU_F = np.array(pd.read_json(os.getenv("VAR_IMU_F")).transpose().iloc[0])
VAR_IMU_W = float(os.getenv("VAR_IMU_W"))
VAR_IMU_M = np.array(pd.read_json(os.getenv("VAR_IMU_M")).transpose().iloc[0])
VAR_GNSS = float(os.getenv("VAR_GNSS"))

GRAVITY = np.array(pd.read_json(os.getenv("GRAVITY")).transpose().iloc[0])

MAG_DIP = float(os.getenv("MAG_DIP")) * np.pi / 180 #[rad]
MAG_LAT = float(os.getenv("MAG_LAT"))
MAG_LNG = float(os.getenv("MAG_LNG"))
THRESHOLD = float(os.getenv("THRESHOLD"))

IS_VAR_TIME_SCALE = eval(os.getenv("IS_VAR_TIME_SCALE"))
IS_EXTERN_LOAD = eval(os.getenv("IS_EXTERN_LOAD"))

DOF = int(os.getenv("DOF"))
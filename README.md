# EKF-Compensation

Extended Kalman Filter 기반의 센서 보정 알고리즘
---
## 환경설정
- python version : 3.9.x
- 라이브러리는 `pip install -r requirements.txt` 로 설치

## Project Directory
```
├── README.md
├── main.py                <- main script used in project
├── requirements.txt       <- library list required
│
├── config                 <- environment varialeScripts used commonly in project
│   └── app_config.py
│    
├── data
│   ├──  __init__.py       <- initialization
│   ├──  data.py           <- Caution! Do not change any part of this script. 
│   ├──  data_management.py   <- data management script 
│   └──  etc (csv, pickle, etec)
│
├── ekf_kosdi_core         <- core of this module
│   ├── __init__.py        <- initialization
│   ├── complementary_filter.py  <- If required, anyone fill out this page
│   └── ekf.py             <- core of extended kalman filter
│   
└── utils                  <- Jupyter notebooks. Naming convention is a number (for ordering),
    ├── __init__.py        <- initialization
    ├── plot_management.py <- (not completed)
    └── rotations.py       <- Caution! Do not change any part of this script. matrix rotation module.
```

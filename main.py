from ekf_kosdi_core.ekf import *
from data.data_management import DataPreProcess
from EKF_Compensation import geomag

def main(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    _ekf = EKF()
    _ekf.ekf_main()
    # gm = geomag.GeoMag()
    # mg = gm.GeoMag(37.498238, 127.0006637)
    # print(mg.dec)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

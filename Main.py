import sys
import os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import time
import random
import csv
import math

from abaqus              import *
from driverUtils         import *
from caeModules          import *
from Simulation3D_Bullet import *


def log(message):
    print(message, file=sys.__stdout__)
    return


#******************
# PLATE PARAMETERS
#******************
PLATE_WIDTH  = 40.0
PLATE_HEIGHT = 2.5


#******************
# BULLET PARAMETERS
#******************
LENGTH_SIDE_RATIO = 3
RADIUS_RANGE      = [2, 3.5]          # [mm]
SPEED_RANGE       = [3000, 10000]     # [mm/s]
BULLET_X_CENTER   = [-PLATE_WIDTH/2, +PLATE_WIDTH/2]
BULLET_Y_CENTER   = [-PLATE_WIDTH/2, +PLATE_WIDTH/2]
BULLET_Z_CENTER   = [-PLATE_WIDTH/2, +PLATE_WIDTH/2]


#******************
# MISCELLANEA
#******************
IDX_START      = 8326
IDX_END        = 8500
INFO_FILE_PATH = "Simulations_Info_" + str(IDX_START) + "_" + str(IDX_END) + ".csv"

with open(file=INFO_FILE_PATH, mode='w', newline='') as info_csv:
    info_csv_writer = csv.writer(info_csv)
    info_csv_writer.writerow([
        "INDEX",
        "SIMULATION_TIME",
        "SIMULATION_LENGTH",
        "COMPLETED",
        "INIT_SPEED",
        "BULLET_RADIUS",
        "BULLET_X_CENTER",
        "BULLET_Y_CENTER",
        "BULLET_Z_CENTER",
        "IDX_FRAME_2/30",
        "IDX_FRAME_1/30",
        "ANGLE_1",
        "ANGLE_2"
    ])


for idx in range(IDX_START, IDX_END + 1):

    log("Simulation " + str(idx))
    start = time.time()

    #******************
    # RANDOM PARAMETERS
    #******************
    radius  = random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1])
    speed   = random.uniform(SPEED_RANGE[0], SPEED_RANGE[1])
    angle1 = random.uniform(0, 30)
    angle2 = random.uniform(0, 30)

    bullet_y_center = (2 / 30) * speed
    bullet_x_center = random.choice([-1, 1]) * bullet_y_center * math.tan(math.radians(angle1))
    bullet_z_center = random.choice([-1, 1]) * bullet_y_center * math.tan(math.radians(angle2))

    #*******************
    # RUNNING SIMULATION
    #*******************
    sim = Simulation3D(
        PLATE_WIDTH=PLATE_WIDTH,
        PLATE_HEIGHT=PLATE_HEIGHT
    )

    simulation_length, simulation_completed, indices = sim.runSimulation(
        BULLET_RADIUS=radius,
        BULLET_SPEED=speed,
        BULLET_X_CENTER=bullet_x_center,
        BULLET_Y_CENTER=bullet_y_center,
        BULLET_Z_CENTER=bullet_z_center,
        SIMULATION_ID=idx,
        LENGTH_SIDE_RATIO=LENGTH_SIDE_RATIO
    )

    simulation_time = str(time.time() - start)

    with open(file=INFO_FILE_PATH, mode='a', newline='') as info_csv:
        info_csv_append = csv.writer(info_csv)
        info_csv_append.writerow([
            idx,
            simulation_time,
            simulation_length,
            simulation_completed,
            speed,
            radius,
            bullet_x_center,
            bullet_y_center,
            bullet_z_center,
            indices[0],
            indices[1],
            angle1,
            angle2
        ])

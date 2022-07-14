# This program creates voxelized 2D images.
# This program requests an input while running. The input should be a path to the folder containing LAS files.
# column, row, depth width should be the same and is stored in the variable "dimension"
# grid[] is a 3D list.
from keras.preprocessing.image import array_to_img
import os
from liblas import file as lasfile
import math
from PIL import Image as im
import numpy as np
from scipy import ndimage, misc
dimension = 0.25
IMG_height = 672 # (224 x 3 = 672) You may uncomment this and the next line if you want fixed height and width for output images rather than the height and width of each canopy
IMG_width = 672 # (224 x 3 = 672) IMG_height and IMG_width should be always the same
text = 'train' #YOU MUST change the string here to 'valid' or 'test' for creating PNG images of corresponding categories from LAS files.
rotations = 360 #YOU MUST change the number of rotations to 1 for 'valid' AND 'test' to avoid any rotations.
## All points in a LAS file needs to be converted into an object of the following class for ease of handling data.
class LASPoint:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.intensity = 0
        self.return_number = 0
        self.number_of_returns = 0
        self.scan_direction = 0
        self.flightline_edge = 0
        self.classification = 0
        self.scan_direction = 0
        self.user_data = 0
        self.point_source_id = 0
        self.height = 0.0
        self.grid_row = -1
        self.grid_column = -1
        self.smoothed_z = 0.0
        self.smoothed_height = 0.0
        self.marked = False

# Function to convert each point in a LAS file (any version) into an object of class LASPoint
def convertToLASPoint(p):
    lp = LASPoint()
    lp.x = p.x * 1
    lp.y = p.y * 1
    lp.z = p.z * 1
    lp.intensity = p.intensity
    lp.return_number = p.return_number
    lp.number_of_returns = p.number_of_returns
    lp.scan_direction = p.scan_direction
    lp.flightline_edge = p.flightline_edge
    lp.classification = p.classification
    lp.scan_angle = p.scan_angle
    lp.user_data = p.user_data
    lp.point_source_id = p.point_source_id
    return lp

def gridCloud(cld, cw):
    minX, minY, minZ, maxX, maxY, maxZ = findBorders(cld)
    colNum = int(math.ceil((maxX - minX) / cw))
    rowNum = int(math.ceil((maxY - minY) / cw))
    depthNum = int(math.ceil((maxZ - minZ) / cw))
    print(depthNum)
    grid = [[[[] for j in range(colNum +1)] for i in range(rowNum + 1)] for k in range(depthNum + 1)]
    # print(np.asarray(grid).shape)
    for p in cld:
        if (p.z <= maxZ and (p.classification == 5 or p.classification == 4 or p.classification == 3)):
            xp = int((p.x - minX) / cw)
            yp = int((p.y - minY) / cw)
            zp = int((p.z - minZ) / cw )
            # print (p.classification)
            grid[zp][yp][xp].append(p)
    return grid, colNum, rowNum, depthNum

def findBorders(cld):
    minX = min(p.x for p in cld) - 0
    minY = min(p.y for p in cld) - 0
    minZ = min(p.z for p in cld) - 0
    maxX = max(p.x for p in cld) + 0
    maxY = max(p.y for p in cld) + 0
    maxZ = max(p.z if (p.classification == 5 or p.classification == 4 or p.classification == 3) else 0 for p in cld) + 0
    # maxZ = max(p.z for p in cld) + 0
    return minX, minY, minZ, maxX, maxY, maxZ

folder_path = input("Enter folder path: ")

for i in os.listdir(folder_path):
    print (i)
    fin = lasfile.File(folder_path + "/" + i, mode='r')
    cloud = []
    # The following loop appends all points within the LAS files into the list 'cloud' as LASPoint class' objects.
    for p in fin:
        cloud.append(convertToLASPoint(p))

    # The gridding function. Return includes 'grid'.
    grid, colNum, rowNum, depthNum = gridCloud(cloud, dimension)
    # depthNum = len(grid)
    #print (depthNum)

    # Creating empty 2D or 3D arrays with zeros
    image1 = [[0 for i in range(0, rowNum)] for j in range(0, depthNum)] # image1 represents side view: yz plane

    for depth in range(1, depthNum):
        for row in range(0, rowNum):
            image1[depth][row] = sum(len(grid[depth][row][col]) for col in range(0, colNum))


    image1_np_array = np.asarray(image1, dtype=np.uint8)


    if not os.path.isdir(folder_path + "/PNGs"):
        os.mkdir(folder_path + "/PNGs")
        print ("Folder created")


    top_pad = int((IMG_height - depthNum)/2)
    bottom_pad = IMG_height - (depthNum + top_pad)
    left_pad_col = int((IMG_width - colNum)/2)
    right_pad_col = int(IMG_width - (colNum + left_pad_col))
    left_pad_row = int((IMG_width - rowNum)/2)
    right_pad_row = int(IMG_width - (rowNum + left_pad_row))

    image1_np_array = np.pad(image1_np_array, [(top_pad, bottom_pad), (left_pad_row, right_pad_row)], mode='constant', constant_values=0)
    image1_np_array = np.expand_dims(image1_np_array, axis = 2)

    imr1 = array_to_img(image1_np_array)

    imr1.save(folder_path + "/PNGs/" + os.path.splitext(i)[0] + "_1" + ".png")

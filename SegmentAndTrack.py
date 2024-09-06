import torch
import pathlib
import warnings
import time
import segmentationTrackFunctions as functions
##Supress warning when saving images
warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")

device = "cuda" if torch.cuda.is_available() else "cpu"

#### BOOLEANS ####
cellposeSegmenation = True
trackCells = True
visualizeTracks = False
GFPFilter = True
haveMasksAlready = False

#### USER INPUTS #####
dataLoc = pathlib.Path(input("Where is your data located at?\n"))
imgList = list(dataLoc.glob("*sm.tif"))

cellPoseModelChoosen = "cyto3"
diameterCellPose = 140
flowThresholdCellPose = 2.19
minSizeCellposeMask = 40
cellprobThreshold = -1
channelsListCellPose = [0,2]
trackastraModel = "greedy_nodiv"
trackastraMaxDistance = 50

print(str(len(imgList)) + " files found! Processing now...")

for img in imgList:
    imgName = img.stem
    print("Now working on...." + str(imgName))
    #let's read the image and break it into channels
    img = imread(img)
    ch0 = img[:,:,:,0]
    ch1 = img[:,:,:,1]
    ch2 = img[:,:,:,2]
    
    masks, cellPoseTime = functions.runCellpose(haveMasksAlready, dataLoc, imgName, cellPoseModelChoosen, img, diameterCellPose, channelsListCellPose, flowThresholdCellPose, minSizeCellposeMask, cellprobThreshold)

    ##now we can track the cells using Trackastra
    masks_tracked, trackTime = functions.runTrackastra(ch0, masks, trackastraModel, trackastraMaxDistance, imgName, device, dataLoc, visualizeTracks, img)
   

    ##now for some basic analysis of the tracked cells!
    analyzeTime = functions.runAnalysis(masks_tracked, ch1, dataLoc, imgName)

    functions.writeParameterFile(dataLoc, imgList, cellPoseModelChoosen, diameterCellPose, flowThresholdCellPose, minSizeCellposeMask, cellprobThreshold, channelsListCellPose, trackastraModel, trackastraMaxDistance)

    print("All done! Here's how long things took: cellpose segmentation took " + str(round(cellPoseTime)) +" seconds. Tracking took: " + str(round(trackTime)) + " seconds. Analysis took: " + str(analyzeTime) + " seconds. Thanks for playing along!")



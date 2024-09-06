import torch
import pathlib
import warnings
import time
from skimage.io import imread, imsave
import segmentationTrackFunctions as functions
import subprocess
 
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
cellPoseModelChoosen = "cyto3"
diameterCellPose = 140
flowThresholdCellPose = 2.19
minSizeCellposeMask = 40
cellprobThreshold = -1
##here we're passing through ch0 [cyotplasm channel] and ch2 [the optional nuclear channel] to aid in segmentation
channelsListCellPose = [0,2]
##tracking model, options are greedy, and greedy_nodiv
trackastraModel = "greedy_nodiv"
##max distance in pixels that the cells are allowed to move
trackastraMaxDistance = 50

dataLoc = pathlib.Path(input("Where is your data located at?\n"))


imgList = list(dataLoc.glob("*sm.tif"))
print(str(len(imgList)) + " files found! Processing now...")

for img in imgList:
    imgName = img.stem
    print("Now working on...." + str(imgName))
    #let's read the image and break it into channels
    img = imread(img)
    ch0 = img[:,:,:,0]
    ch1 = img[:,:,:,1]
    ch2 = img[:,:,:,2]

    ##generate a master save folder that will have all the results for this image

    saveFolderLoc = dataLoc.joinpath(str(imgName) + "_segmentationTrackingResults")
    if not saveFolderLoc.exists():
        print("Making a save folder now..")
        saveFolderLoc.mkdir()
    
    masks, cellPoseTime = functions.runCellpose(haveMasksAlready, dataLoc, imgName, cellPoseModelChoosen, img, diameterCellPose, channelsListCellPose, flowThresholdCellPose, minSizeCellposeMask, cellprobThreshold, saveFolderLoc, ch0)

    # ##now we can track the cells using Trackastra
    masks_tracked, trackTime = functions.runTrackastra(ch0, masks, trackastraModel, trackastraMaxDistance, imgName, device, dataLoc, visualizeTracks, img, saveFolderLoc)
   
    ##next we can run iLastik to segment out the stress granules
    start = time.time()
    functions.runIlastik(saveFolderLoc, ch0, imgName)
    ilastikTime = round(time.time() - start)

    ##now for some basic analysis of the tracked cells!
    analyzeTime = functions.runAnalysis(masks_tracked, ch1, dataLoc, imgName, saveFolderLoc)

    print("All done! Here's how long things took: cellpose segmentation took: " + str(round(cellPoseTime)) +" seconds. Tracking took: " + str(round(trackTime)) + " seconds. iLastik took: "+str(ilastikTime)+" seconds. Analysis took: " + str(analyzeTime) + " seconds. Thanks for playing along!")


functions.writeParameterFile(dataLoc, imgList, cellPoseModelChoosen, diameterCellPose, flowThresholdCellPose, minSizeCellposeMask, cellprobThreshold, channelsListCellPose, trackastraModel, trackastraMaxDistance)



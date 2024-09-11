
 
def runCellpose(haveMasksAlready, dataLoc, imgName, cellPoseModelChoosen, img, diameterCellPose, channelsListCellPose, flowThresholdCellPose, minSizeCellposeMask, cellprobThreshold, saveFolderLoc, ch0):
    from skimage.io import imread, imsave
    import pathlib
    from cellpose import models, io
    import time
    import numpy as np
 
    ##if we have pre-run masks we can use them here...
    if haveMasksAlready:      
        maskLoc = dataLoc.joinpath("cellPoseMasks")
        maskList = list(maskLoc.glob("*_noslice.tif"))      
        mask = imread(maskList[0])
        img_safe = np.zeros(img.shape)
        mask_safe = np.zeros(ch0.shape)
        for timeslice in range(0, mask.shape[0]):
                if np.sum(mask[timeslice, :, :]) == 0:
                    print("blank slice found, removing now")
                else:
                    img_safe[timeslice, :, :,:] = img[timeslice, :,:,:]
                    mask_safe[timeslice, :, :] = mask[timeslice, :,:]
        masks = mask_safe.astype("int")
        
    ##otherwise we'll run cellpose and generate the masks
    else:
        start = time.time()
        ##run cellpose
        #print("img shape is: " + str(img.shape))
        model = models.Cellpose(gpu=True, model_type=cellPoseModelChoosen)
        masks, flows, styles, diams = model.eval(img, diameter = diameterCellPose, channels = channelsListCellPose,
                                            flow_threshold = flowThresholdCellPose, do_3D = False, min_size = minSizeCellposeMask, cellprob_threshold = cellprobThreshold)
        imsave(saveFolderLoc.joinpath(str(imgName)+"_cellposeMaskResults.tif"), masks)
        finish = time.time()
        cellPoseTime = finish - start
        print("Cellpose took..." +str(round(cellPoseTime))+ " seconds")

        return masks, cellPoseTime
    

def runTrackastra(ch0, masks, trackastraModel, trackastraMaxDistance, imgName, device, dataLoc, visualizeTracks, img, saveFolderLoc):
    import time        
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
    from skimage.io import imread, imsave
    import napari

    start = time.time()
    print("loading model...")
    # Load a pretrained model
    model = Trackastra.from_pretrained("general_2d", device=device)
    # Track the cells
    print("tracking now....")
    track_graph = model.track(ch0, masks, mode=trackastraModel, max_distance=trackastraMaxDistance)  # or mode="ilp", or "greedy_nodiv"

    print("Writing cell tracks....")

    # Write to cell tracking challenge format
    outName = saveFolderLoc.joinpath("tracked_"+str(imgName))
    ctc_tracks, masks_tracked = graph_to_ctc(
            track_graph,
            masks,
            outdir=outName,
    )

    imsave(saveFolderLoc.joinpath(str(imgName) + "_tracked_masks.tif"), masks_tracked)
    finish = time.time()

    trackTime = finish - start
    print("Tracking took..." + str(round(trackTime)) + " seconds")

    if visualizeTracks:
        # Visualise in napari if needed
        napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
        v = napari.Viewer()
        v.add_image(img)
        v.add_labels(masks_tracked)
        v.add_tracks(data=napari_tracks, graph=napari_tracks_graph)
        napari.run()


    return masks_tracked, trackTime



def runAnalysis(masks_tracked, ch1, dataLoc, imgName, saveFolderLoc):
    import time
    import pandas as pd
    from alive_progress import alive_bar
    import numpy as np
    from skimage.io import imread, imsave
    from skimage.measure import label, regionprops, regionprops_table

    start = time.time()
    maxValue = round(np.max(masks_tracked))
    print(masks_tracked.shape)
    print(str(maxValue) + "  cells found, analyzing them now...")
    GFPMask = np.zeros(masks_tracked.shape)

    columns = ["cell number","GFP positive", "GFP Value", "frame Number", "area per timestep"]
    dataFrame = pd.DataFrame(columns=columns)

    with alive_bar(maxValue) as bar:
            print("Now scanning for GFP positive cells...")
            for value in range(1, maxValue):
                imgAreaList = []

                boolMask = masks_tracked == value
                filteredGFP = np.where(boolMask, ch1, 0)
                avgGFPValue = np.mean(filteredGFP[filteredGFP>0])  
                gfpBool = False                

                if avgGFPValue > 110:
                        gfpBool = True
                        print("found GFP postive cell, id number: " + str(value))
                        GFPMask += filteredGFP    
                foundZero = False
                ##replace with scikit measuring?
                for tSlice in range(0, filteredGFP.shape[0]):
                        slicecount = np.count_nonzero(filteredGFP[tSlice,:,:]) * (0.183*0.183)
                        if slicecount == 0 and foundZero:
                            ##cell is lost
                            lengthTracked = tSlice + 1
                            foundZero = True
                        if slicecount != 0:
                            imgAreaList.append(slicecount)
                            lengthTracked = tSlice
                        #props = regionprops_table(filteredGFP[tSlice,:,:], properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length'))
                        #propsTable_row = pd.Dataframe(props)
                        new_row = {"cell number": value, "GFP positive": gfpBool, "GFP Value": avgGFPValue, "frame Number": tSlice, "area per timestep": slicecount}
                        dataFrame = pd.concat([dataFrame, pd.DataFrame([new_row])], ignore_index=True)
                avgAreaImg = sum(imgAreaList) / round(len(imgAreaList))
                
                bar()

    analyzeTime = round(time.time() - start)
    dataFrame.to_csv(saveFolderLoc.joinpath(str(imgName)+"_dataframe.csv"))          
    imsave(saveFolderLoc.joinpath(str(imgName) + "_gfpMask.tif"), GFPMask)

    return analyzeTime



def runIlastik(saveFolderLoc, ch0, imgName, iLastikProgramLocation, iLastikFileLocation):
    from pathlib import Path
    import subprocess
    from skimage.io import imread, imsave
    sliceLocation = saveFolderLoc.joinpath("ch1slices")
    if not sliceLocation.exists():
        sliceLocation.mkdir()
    for timestep in range(0, ch0.shape[0]):
         tmpSlice = ch0[timestep,:,:]
         imsave(sliceLocation.joinpath(str(imgName)+"_ch0_"+str(timestep)+".tif"), tmpSlice)

    # assuming you want to process all tiff files in some folder
    in_folder = sliceLocation
    out_folder = sliceLocation.joinpath("stressMasks")
    if not out_folder.exists():
            out_folder.mkdir()

    for in_file in in_folder.glob("*.tif"):
        # generate some output file name
        out_file = out_folder / in_file.name.replace(".tif", "_segmentation.tif")

        print(f"Processing {in_file} -> {out_file}")
        subprocess.run([
        iLastikProgramLocation,
        '--headless',
        f'--project=%s' %iLastikFileLocation,
        '--export_source=simple segmentation',
        f'--raw_data={in_file}',
        f'--output_filename_format={out_file}' 
        ]) 





def writeParameterFile(dataLoc, imgList, cellPoseModelChoosen, diameterCellPose, flowThresholdCellPose, minSizeCellposeMask, cellprobThreshold, channelsListCellPose, trackastraModel, trackastraMaxDistance):    
    from datetime import datetime

    parameterFileLoc = dataLoc.joinpath("classificationParameters.txt")
    counter = 1
    while parameterFileLoc.is_file():
        parameterFileLocStem = parameterFileLoc.stem
        parameterFileLocSuffix = parameterFileLoc.suffix
        parameterFileLoc = dataLoc.joinpath(f"{parameterFileLocStem}_{counter}{parameterFileLocSuffix}")
        counter += 1

    currentDate = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(parameterFileLoc, 'w') as file:
        file.write(f"""File created on: %s
                    
                    Cellpose segmentation + Trackastra was run to segment out the cells and track them as a function of time. The following parameters were used: 

                    Program was run on the following files: 
                    {', '.join(file.stem for file in imgList)}
                    
                    Cellpose model was: %s
                    Cellpose diameter was: %s
                    Cellpose flow threshold was: %s
                    Cellpose minimum size mask was: %s
                    Cellpose probability threshold was: %s
                    Cellpose channels list was: [%s, %s]
                    Trackastra model was: %s
                    Trackastra maximum distance was: %s

        
        """%(currentDate, cellPoseModelChoosen, diameterCellPose, flowThresholdCellPose, minSizeCellposeMask, cellprobThreshold, str(channelsListCellPose[0]), str(channelsListCellPose[1]), trackastraModel, trackastraMaxDistance)
        )

import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

def calibrate(showPics=True):
        # Read Image
        root = os.getcwd()
        calibrationDir = os.path.join(root,'CalibrationImages')
        imgPathList = glob.glob(os.path.join(calibrationDir,'*.jpg'))
        print(f"Looking in: {calibrationDir}")
        print(f"Found: {imgPathList}")

        if len(imgPathList) == 0:
            print("ERROR: No images found in:\n {calibrationDir}")
            return None, None

        # Initialise
        nRows = 8
        nCols = 5
        termCriteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,30,0.001)
        worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
        worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
        worldPtsList = []
        imgPtsList = []

        # Find Corners
        for curImgPath in imgPathList:
            imgBGR = cv.imread(curImgPath)
            if imgBGR is None:
                print(f"  WARNING: Could not read {curImgPath}, skipping")
                continue
            imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
            cornersFound, cornersOrg = cv.findChessboardCorners(
                 imgGray, (nRows, nCols), 
                 flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK)

            if cornersFound == True:
                worldPtsList.append(worldPtsCur)
                cornersRefined = cv.cornerSubPix(
                     imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
                imgPtsList.append(cornersRefined)
                print(f" Corners found: {os.path.basename(curImgPath)}")
                if showPics:
                    cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined,
                    cornersFound)
                    cv.imshow('Chessboard', imgBGR)
                    cv.waitKey(500)
            else:
                 print(f" No corners: {os.path.basename(curImgPath)}")
        cv.destroyAllWindows()

        if len(worldPtsList) == 0:
            print("ERROR: No corners detected in any image. Check nRows/nCols and image quality.")
            return None, None

        # Calibrate
        repError,camMatrix,distCoeff,rvecs,tvecs = cv.calibrateCamera(
             worldPtsList,imgPtsList,imgGray.shape[::-1],None,None)
        print("Camera Matrix:\n", camMatrix)
        print("Reproj Error (pixels): {:.4f}".format(repError))

        # Save Calibration Parameters
        curFolder = os.path.dirname(os.path.abspath(__file__))
        paramPath = os.path.join(curFolder, 'CalibrationResults.npz')
        np.savez(paramPath,
            repError=repError,
            camMatrix=camMatrix,
            distCoeff=distCoeff,
            rvecs=rvecs,
            tvecs=tvecs)
        
        return camMatrix,distCoeff

def runCalibration():
    calibrate(showPics=True)

if __name__ == '__main__':
     # runCalibration()
     runCalibration()
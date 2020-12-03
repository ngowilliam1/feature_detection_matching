import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os
import argparse
import support
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'outlierReject',
        help='Set to True to also do outlier rejection, set to False to only apply epipolar constraint',
        type = str2bool,
        choices=[False, True],
    )
    parser.add_argument(
        '--saveDetectorFeaturesOnTestSet',
        help='Set to True to save detector features from test set ("Saves Features before epipolar constraint")',
        type = str2bool,
        default=False,
    )
    parser.add_argument(
        '--saveTrainDepthResults',
        help='Set to True to save depth results from training set',
        type = str2bool,
        default=False,
    )
    parser.add_argument(
        '--gridSearch',
        help='Set to True to do grid search over distance and ratio threshold',
        type = str2bool,
        default=False,
    )
    parser.add_argument(
        '--distThreshold',
        help='Distance threshold to use ("Only used during outlier detection set to True")',
        type = float,
        default=30,
    )
    parser.add_argument(
        '--ratioThreshold',
        help='Ratio threshold to use ("Only used during outlier detection set to True")',
        type = float,
        default=0.95,
    )
    parser.add_argument(
        '--perImageStatistic',
        help='Print RMSE and number of matches per image on Training Set',
        type = str2bool,
        default= False,
    )

    args = parser.parse_args()

        
    ## Input
    left_image_dir = os.path.abspath('./training/left')
    left_image_dir_test = os.path.abspath('./test/left')
    right_image_dir = os.path.abspath('./training/right')
    right_image_dir_test = os.path.abspath('./test/right')
    calib_dir = os.path.abspath('./training/calib')
    calib_dir_test = os.path.abspath('./test/calib')
    gt_depth_dir = os.path.abspath('./training/gt_depth_map')
    sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
    sample_list_test = ['000011', '000012', '000013', '000014', '000015']

    ## Output
    test_detection_dir = os.path.abspath('./results/detection')
    test_matching_with_rej_dir = os.path.abspath('./results/matching_with_outlier_rej')
    test_matching_without_rej_dir = os.path.abspath('./results/matching_without_outlier_rej')
    Path(test_detection_dir).mkdir(parents=True, exist_ok=True)
    Path(test_matching_with_rej_dir).mkdir(parents=True, exist_ok=True)
    Path(test_matching_without_rej_dir).mkdir(parents=True, exist_ok=True)
    
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=1000)

    # create FlannBasedMatcher object
    flann = cv.FlannBasedMatcher({
        "algorithm": 6,
        "table_number": 6,
        "key_size": 12,
        "multi_probe_level": 2  
    }, {})
    
    
    if args.outlierReject:
        print(f'Detection & Matching WITH outlier rejection started ')
        output_file = open("P3_result.txt", "a")
        output_file.truncate(0)
        if args.gridSearch:
            ratioThreshold = [0.95, 0.9, 0.85]
            distanceThreshold = [70, 60, 50, 40, 30, 20]
        else:
            ratioThreshold = [args.ratioThreshold]
            distanceThreshold = [args.distThreshold]
    else:
        print(f'Detection & Matching WITHOUT outlier rejection started')
        # Ratio and distance threshold will not be used as we do not apply Lowe's ratio test
        ratioThreshold = [args.ratioThreshold]
        distanceThreshold = [args.distThreshold]

    
    ## For Training Set
    for distance in distanceThreshold:
        for ratio in ratioThreshold:
            savedDepth = []
            savedGTDepth = []
            rmseVariance = []
            toKeep = {"imageName":[], "u": [], "v": [], "disparity": [], "depth": [], "GTDepth": [], "dist":[]}

            for sample_name in sample_list:
                
                left_image_path = left_image_dir +'/' + sample_name + '.png'
                right_image_path = right_image_dir +'/' + sample_name + '.png'
                img_left = cv.imread(left_image_path, 0)
                img_right = cv.imread(right_image_path, 0)

                # Feature Detection and Description
                kp1, des1 = orb.detectAndCompute(img_left, None)
                kp2, des2 = orb.detectAndCompute(img_right, None)
                perImageDepth = {"d":[], "gtD":[]}

                # Perform feature matching (without outlier rejection)
                if not args.outlierReject:
                    # Match descriptors.
                    matches = flann.match(des1, des2)
                    
                    # Filter match to be same y axis pixel
                    idxToKeep = support.checkIfYMatches(kp1, kp2, matches)
                    keptMatches = [matches[i] for i in idxToKeep]

                    # Get focal length and baseline
                    train_text_path = calib_dir +'/' + sample_name + '.txt'
                    frame_calib = support.read_frame_calib(train_text_path)
                    calib = support.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
                    baseline = calib.baseline
                    focal = calib.f
                    
                    # Get ground truth depth from lidar
                    gtDepthPath = gt_depth_dir + '/' + sample_name + '.png'
                    imgDepth = cv.imread(gtDepthPath, 0)
                    
                    # Calculate disparity and depth
                    disparity, gtDepth, pixelU, pixelV = support.calculateResults(kp1, kp2, keptMatches, imgDepth)
                    calculatedDepth = list(focal*baseline/disparity)
                    
                    for idx, disp in enumerate(disparity):
                        # Only compare depths if in ground truth the data is available (depth > 0) and remove case where calculataedDepth = inf (ie disparity = 0)
                        if gtDepth[idx] > 0 and disp > 0:
                            if args.saveTrainDepthResults:
                                toKeep["imageName"].append(sample_name + '.png')
                                toKeep["u"].append(pixelU[idx])
                                toKeep["v"].append(pixelV[idx])
                                toKeep["disparity"].append(disp)
                                toKeep["depth"].append(calculatedDepth[idx])
                                toKeep["GTDepth"].append(gtDepth[idx])
                            savedDepth.append(calculatedDepth[idx])
                            savedGTDepth.append(gtDepth[idx])
                            perImageDepth["d"].append(calculatedDepth[idx])
                            perImageDepth["gtD"].append(gtDepth[idx])
                
                # Perform feature matching (with outlier rejection)
                if args.outlierReject:
                    # Match descriptors.
                    matches = flann.knnMatch(des1, des2, k=2)

                    # Apply Lowe's ratio test and distance threshold
                    good = []
                    for m,n in matches:
                        if m.distance < ratio*n.distance and m.distance < distance:
                            good.append(m)
                    matches = good

                    # Filter match to be same y axis pixel
                    idxToKeep = support.checkIfYMatches(kp1, kp2, matches)
                    keptMatches = [matches[i] for i in idxToKeep]

                    # Get focal length and baseline
                    train_text_path = calib_dir +'/' + sample_name + '.txt'
                    frame_calib = support.read_frame_calib(train_text_path)
                    calib = support.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
                    baseline = calib.baseline
                    focal = calib.f
                    
                    gtDepthPath = gt_depth_dir + '/' + sample_name + '.png'
                    imgDepth = cv.imread(gtDepthPath, 0)
                    

                    disparity, gtDepth, pixelU, pixelV = support.calculateResults(kp1, kp2, keptMatches, imgDepth)
                    calculatedDepth = list(focal*baseline/disparity)
                    for idx, disp in enumerate(disparity):
                        # Only compare depths if in ground truth the data is available (depth > 0) and remove case where calculataedDepth = inf (ie disparity = 0)
                        if gtDepth[idx] > 0 and disp > 0:
                            if args.saveTrainDepthResults:
                                toKeep["imageName"].append(sample_name + '.png')
                                toKeep["u"].append(pixelU[idx])
                                toKeep["v"].append(pixelV[idx])
                                toKeep["disparity"].append(disp)
                                toKeep["depth"].append(calculatedDepth[idx])
                                toKeep["GTDepth"].append(gtDepth[idx])
                            savedDepth.append(calculatedDepth[idx])
                            savedGTDepth.append(gtDepth[idx])
                            perImageDepth["d"].append(calculatedDepth[idx])
                            perImageDepth["gtD"].append(gtDepth[idx])

                    if args.perImageStatistic:
                        print(f'RMSE for image {sample_name}: {mean_squared_error(perImageDepth["d"], perImageDepth["gtD"], squared = False):.2f}, Matches: {len(perImageDepth["d"])}')
                
                rmseVariance.append(mean_squared_error(perImageDepth["d"], perImageDepth["gtD"], squared = False))

            if args.outlierReject:         
                print(f"Distance Threshold:{distance}, Ratio Threshold:{ratio:.2f}, Matches: {len(savedDepth)}, RMSE for depth: {mean_squared_error(savedDepth, savedGTDepth, squared = False):.2f}, RMSE std: {np.float(np.std(rmseVariance)):.2f}")
                
    
    if not args.outlierReject:
        print(f"Without outlier rejection, Matches: {len(savedDepth)}, RMSE for depth: {mean_squared_error(savedDepth, savedGTDepth, squared=False):.2f}, RMSE std:{np.float(np.std(rmseVariance)):.2f}")

    if args.saveTrainDepthResults:
        dfDepth = pd.DataFrame.from_dict(toKeep)
        if args.outlierReject:
            dfDepth.to_excel('results/trainDepthResultsWithOutlierDetection.xlsx', index=False)
        else:
            dfDepth.to_excel('results/trainDepthResultsWithoutOutlierDetection.xlsx', index=False)
        
        
    print("Detection & Matching Finished on Training Set")
    
    ## For Test Set
    for sample_name in sample_list_test:
        
        left_image_path = left_image_dir_test +'/' + sample_name + '.png'
        right_image_path = right_image_dir_test +'/' + sample_name + '.png'

        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

        
        # Feature Detection and Description
        kp1, des1 = orb.detectAndCompute(img_left,None)
        kp2, des2 = orb.detectAndCompute(img_right,None)

        # Creates an image with keyPoints on Test set
        img_left_withKP=cv.drawKeypoints(img_left,kp1,img_left)
        cv.imwrite(test_detection_dir + '/' + f'{sample_name}_left.jpg',img_left_withKP)
        

        # Creates a dataframe to later create a csv containing all descriptors for all keypoints in image
        if args.saveDetectorFeaturesOnTestSet:
            toKeep = {"left-right":[], "imageName": [], "x": [], "y": [], "feat": []}
            for idx,kp in enumerate(kp1):
                toKeep["left-right"].append("left")
                toKeep["imageName"].append(sample_name + '.png')
                toKeep["x"].append(int(kp.pt[0]))
                toKeep["y"].append(int(kp.pt[1]))
                toKeep["feat"].append(list(des1[idx]))
            for idx,kp in enumerate(kp2):
                toKeep["left-right"].append("right")
                toKeep["imageName"].append(sample_name + '.png')
                toKeep["x"].append(int(kp.pt[0]))
                toKeep["y"].append(int(kp.pt[1]))
                toKeep["feat"].append(list(des2[idx]))
            
        
        # Perform feature matching (without outlier rejection)
        if not args.outlierReject:
            # Match descriptors.
            matches = flann.match(des1,des2)
            # Filter match to be same y axis pixel
            idxToKeep = support.checkIfYMatches(kp1, kp2, matches)
            keptMatches = [matches[i] for i in idxToKeep]

            img = cv.drawMatches(img_left, kp1, img_right, kp2, keptMatches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(test_matching_without_rej_dir + '/' + f'{sample_name}.jpg', img)

        # Perform feature matching (with outlier rejection)
        if args.outlierReject:

            # Match descriptors.
            matches = flann.knnMatch(des1, des2, k=2)
            # Apply ratio test
            matchesPassingRatioTest = []
            for m,n in matches:
                if m.distance < args.ratioThreshold*n.distance and m.distance < args.distThreshold:
                    matchesPassingRatioTest.append(m)

            idxToKeep = support.checkIfYMatches(kp1, kp2, matchesPassingRatioTest)
            keptMatches = [matchesPassingRatioTest[i] for i in idxToKeep]

            test_text_path = calib_dir_test +'/' + sample_name + '.txt'
            frame_calib = support.read_frame_calib(test_text_path)
            calib = support.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
            baseline = calib.baseline
            focal = calib.f

            disparity, _, pixelU, pixelV = support.calculateResults(kp1, kp2, keptMatches)
            calculatedDepth = list(focal*baseline/disparity)
            
            # TODO: To test this following output
            # Output
            for u, v, disp, depth in zip(pixelU, pixelV, disparity, calculatedDepth):
                line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
                output_file.write(line + '\n')

            # Draw matches
            img = cv.drawMatches(img_left, kp1, img_right, kp2, keptMatches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(test_matching_with_rej_dir + '/' + f'{sample_name}.jpg', img)

    if args.saveDetectorFeaturesOnTestSet:
        dfTest = pd.DataFrame.from_dict(toKeep)
        dfTest.to_excel('results/detectionResultsTestSet.xlsx', index=False)
    if args.outlierReject:
        output_file.close()
    print("Detection & Matching Finished on Test Set")
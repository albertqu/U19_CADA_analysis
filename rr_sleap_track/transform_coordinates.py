import cv2
import numpy as np
import pandas as pd
import os
from h5_preprocessing import h5_preprocessing

def get_warp_matrix(filename, logger):
    '''
    Define warp matrix by the key points coordinates according to file names (camera location).
    3 key points are 3 corners of the restaurants.
    '''
    try:
        if 'R1' in str(filename):
            srcTri = np.array([[186.9, 178.3], [226, 178.3], [224, 137.6]]).astype(np.float32)
        elif 'R2' in str(filename):
            srcTri = np.array([[271.2, 128.3], [270.8, 95.6], [228.7, 95.3]]).astype(np.float32)
        elif 'R3' in str(filename):
            srcTri = np.array([[208.5, 53.5], [170, 54.5], [170.7, 89.5]]).astype(np.float32)
        elif 'R4' in str(filename):
            srcTri = np.array([[108.4, 119.3], [109.1, 158.5], [155.3, 159.2]]).astype(np.float32)
        else:
            logger.error(f"Cannot find warp matrix for this file: {filename}")
            return None

        dstTri = np.array([[350, 46], [350, 10], [309, 10]]).astype(np.float32)
        warp_matrix = cv2.getAffineTransform(srcTri, dstTri)
        return warp_matrix
    except Exception as e:
        logger.error(f"Error calculating warp matrix for {filename}: {e}")
        raise

def warp_coordinates(h5_file, logger):
    '''
    Apply affine transformation to the input HDF5 file to align with R2 camera orientation
    
    Return:
    a pandas DataFrame
    '''
    try:
        logger.info(f"Starting transformation for: {h5_file.split('/')[-1]}")

        df = h5_preprocessing(h5_file) 

        # Define the columns containing the coordinates
        coordinate_columns = [
            'Head x', 'Head y',
            'Neck x', 'Neck y',
            'Torso x', 'Torso y',
            'Tailhead x', 'Tailhead y'
        ]

        warped_columns = [
            'warped Head x', 'warped Head y',
            'warped Neck x', 'warped Neck y',
            'warped Torso x', 'warped Torso y',
            'warped Tailhead x', 'warped Tailhead y'
        ]

        # Extract the coordinates
        coords = df[coordinate_columns].values
        warp_matrix = get_warp_matrix(h5_file.split('/')[-1], logger)

        # Apply affine transformation to each pair of coordinates
        transformed_coords = np.empty_like(coords)
        for i in range(0, coords.shape[1], 2): 
            points = coords[:, i:i+2]
            points = points.reshape(-1, 1, 2)
            transformed_points = cv2.transform(points, warp_matrix).reshape(-1, 2)
            transformed_coords[:, i:i+2] = transformed_points

        # Update the dataframe with the transformed and cropped coordinates
        df[warped_columns] = transformed_coords
        
        logger.info(f"Transformation complete for file: {h5_file.split('/')[-1]}")
        return df

    except Exception as e:
        logger.error(f"Error processing {h5_file.split('/')[-1]}: {e}")
        raise   


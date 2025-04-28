import h5py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))
    
    # Interpolate along each column
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant
        x = np.flatnonzero(~np.isnan(y))
        if len(x) > 1:  # Ensure there are enough points to interpolate
            f = interp1d(x, y[x], kind=kind, fill_value="extrapolate", bounds_error=False)
            
            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)

            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(y)
            if mask.any():
                y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        
        # Save slice
        Y[:, i] = y
    
    # Restore to initial shape
    Y = Y.reshape(initial_shape)
    return Y

def h5_preprocessing(input_h5):
    '''
    Convert HDF5 file into a DataFrame, then fill missing values using interpolation.
    '''
    with h5py.File(input_h5, 'r') as f:
        tracks_matrix = f["tracks"][:].transpose()
        nodes = [n.decode() for n in f["node_names"][:]]

    def coordinate(node_index, x_or_y):
        """Generate coordinate arrays for x or y based on node index."""
        coordinates = np.array([])
        for i in range(tracks_matrix.shape[0]):
            if x_or_y == "x":
                coordinates = np.append(coordinates, tracks_matrix[i, node_index, 0, 0])
            elif x_or_y == "y":
                coordinates = np.append(coordinates, tracks_matrix[i, node_index, 1, 0])
        return coordinates

    ihead = 0
    ineck = 1
    itorso = 2
    itailhead = 3

    head_x_coordinates = coordinate(ihead, "x")
    head_y_coordinates = coordinate(ihead, "y")
    neck_x_coordinates = coordinate(ineck, "x")
    neck_y_coordinates = coordinate(ineck, "y")
    torso_x_coordinates = coordinate(itorso, "x")
    torso_y_coordinates = coordinate(itorso, "y")
    tailhead_x_coordinates = coordinate(itailhead, "x")
    tailhead_y_coordinates = coordinate(itailhead, "y")

    df = pd.DataFrame(
        {
            "Head x": head_x_coordinates,
            "Head y": head_y_coordinates,
            "Neck x": neck_x_coordinates,
            "Neck y": neck_y_coordinates,
            "Torso x": torso_x_coordinates,
            "Torso y": torso_y_coordinates,
            "Tailhead x": tailhead_x_coordinates,
            "Tailhead y": tailhead_y_coordinates,
        }
    )

    # Fill missing values using interpolation
    df_filled = pd.DataFrame(fill_missing(df.values, kind="linear"), columns=df.columns)
    
    return df_filled

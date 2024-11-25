import numpy as np
import cupy as cp
from scipy.ndimage import gaussian_filter, convolve1d, median_filter
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import IncrementalPCA
from cuml.decomposition import PCA as cuPCA
from cuml.preprocessing import MinMaxScaler as cuMinMaxScaler, StandardScaler as cuStandardScaler
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from concurrent.futures import ThreadPoolExecutor

class Preprocessor:
    """
    A modular preprocessing pipeline for 3D position data, optimized for memory and computational efficiency.
    """
    def __init__(self, methods=None, use_gpu=False):
        self.methods = methods if methods else []
        self.use_gpu = use_gpu

    def add_method(self, method, params=None):
        """
        Add a preprocessing method to the pipeline.
        """
        self.methods.append((method, params if params else {}))

    def apply(self, data):
        """
        Apply the preprocessing methods sequentially on the data.
        """
        for method, params in self.methods:
            data = method(data, **params)
        return data


# Optimized Preprocessing Methods with GPU Integration

def moving_average_optimized(data, window_size=5):
    """
    Apply an optimized moving average filter to the data using a sliding window.
    """
    result = np.zeros_like(data)
    for i in range(window_size, len(data)):
        result[i] = np.mean(data[i - window_size:i], axis=0)
    return result

def separable_gaussian_filter(data, sigma=1):
    """
    Apply a separable Gaussian filter to reduce computational complexity.
    """
    kernel = np.exp(-np.arange(-3*sigma, 3*sigma+1)**2 / (2*sigma**2))
    kernel /= kernel.sum()
    smoothed = convolve1d(data, kernel, axis=0)
    return smoothed

def batch_process(data, batch_size=100000):
    """
    Process data in batches to prevent memory overload.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def incremental_pca(data, n_components=2, batch_size=100000):
    """
    Apply Incremental PCA for large datasets that don't fit in memory all at once.
    """
    ipca = IncrementalPCA(n_components=n_components)
    for batch in batch_process(data, batch_size):
        ipca.partial_fit(batch)
    return ipca.transform(data)

def min_max_normalize(data):
    """
    Normalize data to a [0, 1] range using cuML (GPU accelerated).
    """
    data = cp.asarray(data, dtype=cp.float32)  # Ensure GPU compatibility
    scaler = cuMinMaxScaler()
    return scaler.fit_transform(data).get()  # Convert back to CPU array after transformation

def z_score_normalize(data):
    """
    Standardize data to have a mean of 0 and a standard deviation of 1 using cuML.
    """
    data = cp.asarray(data, dtype=cp.float32)  # Ensure GPU compatibility
    scaler = cuStandardScaler()
    return scaler.fit_transform(data).get()  # Convert back to CPU array after transformation

def apply_pca(data, n_components=2):
    """
    Apply PCA using cuML for GPU acceleration.
    """
    data = cp.asarray(data, dtype=cp.float32)  # Ensure GPU compatibility
    pca = cuPCA(n_components=n_components)
    return pca.fit_transform(data).get()  # Convert back to CPU array after transformation

def interpolate_missing(data, method='linear'):
    """
    Interpolate missing data points in 3D position data.
    """
    for dim in range(data.shape[1]):
        mask = np.isnan(data[:, dim])
        indices = np.arange(len(data))
        if np.any(mask):
            if method == 'linear':
                interpolator = interp1d(indices[~mask], data[~mask, dim], kind='linear', fill_value='extrapolate')
            elif method == 'spline':
                interpolator = UnivariateSpline(indices[~mask], data[~mask, dim], s=0)
            data[mask, dim] = interpolator(indices[mask])
    return data

def normalize_positions(data):
    """
    Translate 3D positions to the origin (center the data).
    """
    centroid = np.mean(data, axis=0)
    return data - centroid

def rotate_coordinates(data, axis, angle):
    """
    Rotate 3D coordinates around a given axis.
    """
    rotation_matrix = {
        'x': R.from_euler('x', angle, degrees=True).as_matrix(),
        'y': R.from_euler('y', angle, degrees=True).as_matrix(),
        'z': R.from_euler('z', angle, degrees=True).as_matrix()
    }.get(axis)
    
    if rotation_matrix is None:
        raise ValueError("Invalid axis. Choose from 'x', 'y', 'z'.")
    
    return np.dot(data, rotation_matrix.T)

def smooth_trajectory(data, method='gaussian', **kwargs):
    """
    Smooth a 3D trajectory using specified smoothing method.
    """
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1)
        return separable_gaussian_filter(data, sigma=sigma)
    elif method == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        return moving_average_optimized(data, window_size=window_size)
    else:
        raise ValueError("Invalid smoothing method. Choose 'gaussian' or 'moving_average'.")


# Example Usage with Optimizations

# Create Preprocessor
methods = [
    (min_max_normalize, {}),
    (apply_pca, {'n_components': 3}),
    (smooth_trajectory, {'method': 'gaussian', 'sigma': 2}),
]
preprocessor = Preprocessor(methods, use_gpu=True)

# Example 3D position data for nanobots
data = np.random.rand(1000000, 3)  # 1 million data points (3D positions)

# Apply preprocessing methods
processed_data = preprocessor.apply(data)
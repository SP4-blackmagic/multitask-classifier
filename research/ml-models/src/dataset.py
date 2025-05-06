"""Dataset module for handling data loading and preprocessing."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import spectral
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
import warnings

class SpectralDataset:
    """Class for handling spectral dataset operations."""
    
    def __init__(self, config: Dict):
        """Initialize the dataset handler.
        
        Args:
            config: Configuration dictionary containing dataset settings
        """
        self.config = config
        self.root_dir = config['root_dir']
        self.annotations_dir = os.path.join(self.root_dir, config['annotations']['path'])
        self.label_encoders = {}
        self.scaler = None
        self.pca_transformer = None
        
    def load_annotations(self, split: str) -> Optional[Dict]:
        """Load annotations for a specific split.
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            Dictionary containing annotations or None if loading fails
        """
        split_file = self.config['annotations'][f'{split}_file']
        annotations_path = os.path.join(self.annotations_dir, split_file)
        
        if not os.path.exists(annotations_path):
            warnings.warn(f"Annotations file not found: {annotations_path}")
            return None
            
        try:
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            warnings.warn(f"Failed to load annotations: {e}")
            return None
            
    def prepare_split_data(self, annotations_data: Dict, split_name: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Prepare data for a specific split.
        
        Args:
            annotations_data: Dictionary containing annotations
            split_name: Name of the split
            
        Returns:
            Tuple of (DataFrame with prepared data, annotations dictionary)
        """
        if not annotations_data or 'records' not in annotations_data:
            return None, None
            
        try:
            records_df = pd.DataFrame(annotations_data['records'])
            labels_df = pd.DataFrame(annotations_data.get('annotations', []))
            
            # Merge records with labels
            records_df['record_id'] = records_df['id']
            if not labels_df.empty and 'record_id' in labels_df.columns:
                merged_df = pd.merge(records_df, labels_df, on='record_id', how='left', suffixes=('', '_label'))
            else:
                merged_df = records_df.copy()
                
            # Process categorical labels
            for task, col in self.config['tasks'].items():
                if col in merged_df.columns:
                    merged_df[col] = pd.Categorical(merged_df[col])
                    
            # Process numerical labels
            if 'firmness' in merged_df.columns:
                bins = [-np.inf, 1000, 1500, np.inf]
                labels = ['too_soft', 'perfect', 'too_hard']
                merged_df['firmness_category'] = pd.Categorical(
                    pd.cut(merged_df['firmness'], bins=bins, labels=labels, right=True),
                    categories=labels,
                    ordered=False
                )
                
            # Extract fruit ID
            merged_df['fruit_id'] = merged_df['files'].apply(
                lambda x: self._extract_fruit_id(x.get('header_file')) if isinstance(x, dict) else None
            )
            merged_df['fruit_id'] = merged_df['fruit_id'].fillna(merged_df['id'])
            
            return merged_df, annotations_data
            
        except Exception as e:
            warnings.warn(f"Failed to prepare split data: {e}")
            return None, None
            
    def _extract_fruit_id(self, filepath: str) -> Optional[str]:
        """Extract fruit ID from filepath.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Fruit ID or None if extraction fails
        """
        if not isinstance(filepath, str) or not filepath:
            return None
        try:
            base = os.path.basename(filepath)
            parts = base.split('.')[0].split('_')
            return '_'.join(parts[:4]) if len(parts) >= 4 else base.split('.')[0]
        except Exception:
            return filepath
            
    def extract_features(self, cube: np.ndarray, wavelengths: np.ndarray, rec_id: str) -> Optional[np.ndarray]:
        """Extract features from spectral cube.
        
        Args:
            cube: Spectral cube data
            wavelengths: Wavelength values
            rec_id: Record ID
            
        Returns:
            Extracted features or None if extraction fails
        """
        if not isinstance(cube, np.ndarray) or not isinstance(wavelengths, np.ndarray):
            return None
            
        # Remove background
        masked_cube, mask = self._remove_background(cube)
        if masked_cube is None or mask is None or mask.sum() == 0:
            return None
            
        # Extract average spectrum
        fruit_pixels = masked_cube[mask]
        if fruit_pixels.shape[0] == 0:
            return None
            
        try:
            avg_spec = np.mean(fruit_pixels, axis=0, dtype=np.float64)
        except Exception:
            return None
            
        # Extract features
        features_list = []
        
        # Statistical features
        if self.config['features']['use_stat_features']:
            stat_features = self._extract_statistical_features(avg_spec)
            features_list.extend(stat_features)
            
        # Spectral features
        if self.config['features']['use_avg_spectrum']:
            features_list.append(avg_spec)
            
        # Derivatives
        if self.config['features']['use_derivative_1'] or self.config['features']['use_derivative_2']:
            deriv1, deriv2 = self._calculate_derivatives(avg_spec, wavelengths)
            if self.config['features']['use_derivative_1'] and deriv1 is not None:
                features_list.append(deriv1)
            if self.config['features']['use_derivative_2'] and deriv2 is not None:
                features_list.append(deriv2)
                
        # Continuum removal
        if self.config['features']['use_continuum_removed']:
            cr_spec = self._calculate_continuum_removal(avg_spec, wavelengths)
            if cr_spec is not None:
                features_list.append(cr_spec)
                
        # FFT features
        if self.config['features']['use_fft_feature']:
            fft_spec = self._calculate_fft_feature(masked_cube)
            if fft_spec is not None:
                features_list.append(fft_spec)
                
        if not features_list:
            return None
            
        try:
            return np.concatenate(features_list)
        except ValueError:
            return None
            
    def _remove_background(self, cube: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Remove background from spectral cube.
        
        Args:
            cube: Spectral cube data
            
        Returns:
            Tuple of (masked cube, mask)
        """
        if not isinstance(cube, np.ndarray) or cube.ndim != 3:
            return None, None
            
        threshold = self.config['processing']['background_threshold']
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_intensity = np.mean(cube, axis=-1, dtype=np.float64)
            mean_intensity[np.isnan(mean_intensity)] = 0
            
        mask = mean_intensity > threshold
        masked_cube = cube * mask[..., np.newaxis]
        
        return masked_cube, mask
        
    def _extract_statistical_features(self, spectrum: np.ndarray) -> List[float]:
        """Extract statistical features from spectrum.
        
        Args:
            spectrum: Spectral data
            
        Returns:
            List of statistical features
        """
        spec_std = np.std(spectrum)
        if spec_std > 1e-6:
            return [spec_std, skew(spectrum), kurtosis(spectrum)]
        return [spec_std, 0.0, -3.0]
        
    def _calculate_derivatives(self, spectrum: np.ndarray, wavelengths: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate derivatives of spectrum.
        
        Args:
            spectrum: Spectral data
            wavelengths: Wavelength values
            
        Returns:
            Tuple of (first derivative, second derivative)
        """
        if not isinstance(spectrum, np.ndarray) or spectrum.ndim != 1:
            return None, None
            
        window = self.config['features']['savitzky_golay']['window']
        polyorder = self.config['features']['savitzky_golay']['polyorder']
        
        if window % 2 == 0:
            window += 1
        if window > len(spectrum):
            window = len(spectrum) - (1 if len(spectrum) % 2 == 0 else 2)
        if window < 3:
            return None, None
        if polyorder >= window:
            polyorder = window - 1
        if polyorder < 1:
            return None, None
            
        try:
            delta = np.mean(np.diff(wavelengths)) if len(wavelengths) > 1 else 1.0
            deriv1 = savgol_filter(spectrum, window, polyorder, deriv=1, delta=delta)
            deriv2 = savgol_filter(spectrum, window, polyorder, deriv=2, delta=delta)
            return deriv1, deriv2
        except Exception:
            return None, None
            
    def _calculate_continuum_removal(self, spectrum: np.ndarray, wavelengths: np.ndarray) -> Optional[np.ndarray]:
        """Calculate continuum removal from spectrum.
        
        Args:
            spectrum: Spectral data
            wavelengths: Wavelength values
            
        Returns:
            Continuum removed spectrum or None if calculation fails
        """
        if not isinstance(spectrum, np.ndarray) or spectrum.ndim != 1 or len(spectrum) < 3:
            return None
            
        try:
            sort_indices = np.argsort(wavelengths)
            wavelengths_sorted = wavelengths[sort_indices]
            spectrum_sorted = spectrum[sort_indices]
            
            points = np.vstack((wavelengths_sorted, spectrum_sorted)).T
            hull = ConvexHull(points)
            
            upper_hull_indices = []
            start_idx = np.where(hull.vertices == 0)[0][0]
            current_idx = start_idx
            
            while True:
                current_point = hull.vertices[current_idx]
                upper_hull_indices.append(current_point)
                next_idx = (current_idx + 1) % len(hull.vertices)
                next_point = hull.vertices[next_idx]
                
                if current_point == len(points) - 1:
                    break
                if next_point == upper_hull_indices[0] and len(upper_hull_indices) > 1:
                    break
                if len(upper_hull_indices) > len(points):
                    return None
                    
                current_idx = next_idx
                
            upper_hull_indices = np.unique(upper_hull_indices)
            if len(upper_hull_indices) < 2:
                return None
                
            upper_hull_points = points[upper_hull_indices]
            upper_hull_points = upper_hull_points[np.argsort(upper_hull_points[:, 0])]
            
            continuum = np.interp(wavelengths_sorted, upper_hull_points[:, 0], upper_hull_points[:, 1])
            continuum[continuum < 1e-9] = 1e-9
            
            cr_spectrum_sorted = spectrum_sorted / continuum
            cr_spectrum = np.zeros_like(spectrum)
            cr_spectrum[sort_indices] = cr_spectrum_sorted
            
            return cr_spectrum
            
        except Exception:
            return None
            
    def _calculate_fft_feature(self, masked_cube: np.ndarray) -> Optional[np.ndarray]:
        """Calculate FFT features from masked cube.
        
        Args:
            masked_cube: Masked spectral cube
            
        Returns:
            FFT features or None if calculation fails
        """
        if not isinstance(masked_cube, np.ndarray) or masked_cube.ndim != 3:
            return None
            
        try:
            fft_bands = [np.fft.fftshift(np.fft.fft2(masked_cube[:, :, i])) for i in range(masked_cube.shape[2])]
            fft_mags = [np.mean(np.log1p(np.abs(fft_band))) for fft_band in fft_bands]
            return np.array(fft_mags)
        except Exception:
            return None
            
    def fit_transformers(self, X: np.ndarray) -> None:
        """Fit scaler and PCA transformer.
        
        Args:
            X: Feature matrix
        """
        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
        # Fit PCA if enabled
        if self.config['features']['use_pca_features']:
            n_components = self.config['features']['n_pca_components']
            if n_components < X.shape[1]:
                self.pca_transformer = PCA(n_components=n_components)
                self.pca_transformer.fit(X)
                
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted transformers.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed features
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
            
        if self.pca_transformer is not None:
            X_pca = self.pca_transformer.transform(X)
            X = np.concatenate([X, X_pca], axis=1)
            
        return X
        
    def save_transformers(self, output_dir: str) -> None:
        """Save fitted transformers.
        
        Args:
            output_dir: Directory to save transformers
        """
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(output_dir, 'feature_scaler.joblib'))
            
        if self.pca_transformer is not None:
            joblib.dump(self.pca_transformer, os.path.join(output_dir, 'feature_pca.joblib'))
            
    def load_transformers(self, input_dir: str) -> None:
        """Load saved transformers.
        
        Args:
            input_dir: Directory containing saved transformers
        """
        scaler_path = os.path.join(input_dir, 'feature_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        pca_path = os.path.join(input_dir, 'feature_pca.joblib')
        if os.path.exists(pca_path):
            self.pca_transformer = joblib.load(pca_path)

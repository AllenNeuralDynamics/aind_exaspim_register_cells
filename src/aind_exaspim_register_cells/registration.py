"""Registration pipeline for transforming exaSPIM cell data to CCF space.

This module provides the main registration pipeline for transforming
points from image space to CCF space using registration transforms.
"""

import json
import os
from typing import List

import ants
import numpy as np
import pandas as pd
from allensdk.core.swc import Compartment, Morphology
import SimpleITK as sitk

from .utils import CoordinateConverter, OrientationUtils
from .visualization import ImageVisualizer

class RegistrationPipeline:
    """
    Main pipeline for transforming points from image space to CCF space using registration transforms.
    """
    def __init__(self, 
        dataset_id: str, 
        output_dir: str, 
        acquisition_file: str, 
        brain_path: str, 
        resampled_brain_path: str, 
        brain_to_exaspim_transform_path: list,
        exaspim_to_ccf_transform_path: list, 
        ccf_path: str, 
        exaspim_template_path: str, 
        transform_res: list, 
        level: int,
        manual_transform_path: list = None):
        self.dataset_id = dataset_id
        self.output_dir = output_dir
        self.acquisition_file = acquisition_file
        self.brain_path = brain_path
        self.resampled_brain_path = resampled_brain_path
        self.brain_to_exaspim_transform_path = brain_to_exaspim_transform_path
        self.exaspim_to_ccf_transform_path = exaspim_to_ccf_transform_path
        self.ccf_path = ccf_path
        self.exaspim_template_path = exaspim_template_path
        self.transform_res = transform_res
        self.level = level
        self.manual_transform_path = manual_transform_path if manual_transform_path is not None else []


    def load_images(self) -> tuple:
        """
        Load and preprocess CCF, exaspim template, brain, and resampled brain images.

        Returns
        -------
        tuple
            (ccf, ants_exaspim, brain_img, resampled_img)
        """
        ccf = ants.image_read(self.ccf_path)
        ccf = ImageVisualizer.perc_normalization(ccf)
        ants_exaspim = ants.image_read(self.exaspim_template_path)
        ants_exaspim.set_spacing(ccf.spacing)
        ants_exaspim.set_origin(ccf.origin)
        ants_exaspim.set_direction(ccf.direction)
        brain_img = ants.image_read(self.brain_path)
        resampled_img = ants.image_read(self.resampled_brain_path)
        return ccf, ants_exaspim, brain_img, resampled_img

    def preprocess_coords(self, coords: np.ndarray, input_img: np.ndarray, resampled_img: np.ndarray, cell_filename: str = None) -> np.ndarray:
        """
        Preprocess coordinates: convert to index, check orientation, and resample to isotropic.

        Parameters
        ----------
        coords : np.ndarray
            Soma coordinates in physical space.
        input_img : np.ndarray
            Reference image for orientation.
        resampled_img : np.ndarray
            Reference resampled image.
        Returns
        -------
        np.ndarray
            Preprocessed coordinates.
        """
        if self.level == 3:
            scale = [0.02025/0.025, 0.02025/0.025, 0.027/0.025] # xyz
        elif self.level == 2:
            scale = [0.0101/0.01, 0.0101/0.01, 0.0135/0.01] #  xyz 1.08, 0.812, 0.812
        else:
            raise ValueError(f"Level {self.level} not supported, only support 2 and 3")

        soma_locations = self.soma_physical_to_index(coords, self.level+3)
        
        oriented_cells = self.check_orientation(self.acquisition_file, soma_locations, input_img)
        input_img_norm = ImageVisualizer.perc_normalization(input_img)
        if cell_filename:
            ImageVisualizer.soma_overlay_volumn(oriented_cells, input_img_norm, title=f"{cell_filename}_in_raw_data_space", figpath=f"{self.output_dir}/{cell_filename}_in_raw_data_space")
        
        resampled_cells = self.resample_isotropic(oriented_cells, scale)
        resampled_img_norm = ImageVisualizer.perc_normalization(resampled_img)
        if cell_filename:
            ImageVisualizer.soma_overlay_volumn(resampled_cells, resampled_img_norm, title=f"{cell_filename}_in_resampled_space", figpath=f"{self.output_dir}/{cell_filename}_in_resampled_space")
        return resampled_cells

    def apply_transforms_to_points(
        self, 
        resampled_cells: np.ndarray, 
        resampled_img: ants.ANTsImage, 
        ants_exaspim: ants.ANTsImage, 
        ccf: ants.ANTsImage,
        cell_filename: str = None
    ) -> np.ndarray:
        """
        Apply registration transforms to points and return final CCF index coordinates.

        Parameters
        ----------
        resampled_cells : np.ndarray
            Preprocessed soma coordinates.
        resampled_img : ants.ANTsImage
            Resampled brain image.
        ants_exaspim : ants.ANTsImage
            Exaspim template image.
        ccf : ants.ANTsImage
            CCF template image.

        Returns
        -------
        np.ndarray
            Final CCF index coordinates.
        """
       # register to exaspim template
        print("Register to exaspim template ...")
        ants_pts = CoordinateConverter.index_to_physical(resampled_img, resampled_cells)
        df = pd.DataFrame(ants_pts, columns=["x", "y", "z"])
        # Dynamically set whichtoinvert based on the number of transforms
        if len(self.brain_to_exaspim_transform_path) == 1:
            whichtoinvert = [True]
        else:
            whichtoinvert = [True, False]
        ants_pts = ants.apply_transforms_to_points(
            3, df, self.brain_to_exaspim_transform_path, whichtoinvert=whichtoinvert
        )
        ants_pts_exaspim = np.array(ants_pts)
        idx_pts = CoordinateConverter.physical_to_index(ants_exaspim, ants_pts_exaspim)
        if cell_filename:
            ImageVisualizer.soma_overlay_volumn(idx_pts, ants_exaspim.numpy(), title=f"{cell_filename}_in_exaspim_temp_space", figpath=f"{self.output_dir}/{cell_filename}_in_exaspim_temp_space")
        
        # register to ccf
        print("Register to CCF ...")
        df = pd.DataFrame(ants_pts_exaspim, columns=["x", "y", "z"])
        if self.exaspim_to_ccf_transform_path:
            # Only apply transform if the list is not empty
            if len(self.exaspim_to_ccf_transform_path) == 1:
                whichtoinvert_ccf = [True]
            else:
                whichtoinvert_ccf = [True, False]
            ants_pts = ants.apply_transforms_to_points(
                3, df, self.exaspim_to_ccf_transform_path, whichtoinvert=whichtoinvert_ccf
            )
            ants_pts_ccf = np.array(ants_pts)
        else:
            # If no transform, just use the input points
            ants_pts_ccf = np.array(df)
        idx_pts = CoordinateConverter.physical_to_index(ccf, ants_pts_ccf)
        if cell_filename:
            ImageVisualizer.soma_overlay_volumn(idx_pts, ccf.numpy(),title=f"{cell_filename}_in_ccf_space", figpath=f"{self.output_dir}/{cell_filename}_in_ccf_space")
            
            
        # manual registration 
        # Check if any transform is in .nrrd format
        df_manual = pd.DataFrame(ants_pts_ccf, columns=["x", "y", "z"])
        nrrd_transforms = [t for t in self.manual_transform_path if t.endswith('.nrrd')]

        # Apply SimpleITK transforms for .nrrd files
        if nrrd_transforms:
            print("Apply manual transfrom ...")
            for nrrd_transform in nrrd_transforms:
                df_manual = self.apply_simpleitk_transform(nrrd_transform, df_manual)
        ants_pts_ccf = np.array(df_manual)
        idx_pts = CoordinateConverter.physical_to_index(ccf, ants_pts_ccf)
        if cell_filename:
            ImageVisualizer.soma_overlay_volumn(idx_pts, ccf.numpy(),title=f"{cell_filename}_in_ccf_space_manual", figpath=f"{self.output_dir}/{cell_filename}_in_ccf_space_manual")

        return idx_pts, ants_pts_ccf
                

    def check_orientation(
        self, 
        acquisition_path: str, 
        soma_locations: np.ndarray, 
        input_img: np.ndarray,
        show: bool = False
    ) -> np.ndarray:
        """
        Adjust soma locations based on acquisition metadata orientation.

        Parameters
        ----------
        acquisition_path : str
            Path to acquisition metadata JSON.
        soma_locations : np.ndarray
            Soma locations to adjust.
        input_img : np.ndarray
            Reference image for shape.

        Returns
        -------
        np.ndarray
            Oriented soma locations.
        """
        with open(acquisition_path, "r") as f:
            metadata = json.load(f)
            file_name_1st = metadata["tiles"][0]["file_name"]
            if "tile_000000_ch_" in file_name_1st:
                # print("The input is a Beta scope sample!!")
                CCF_DIRECTIONS = {
                    0: "Anterior_to_posterior",
                    1: "Superior_to_inferior",
                    2: "Left_to_right",
                }
            else:
                # print("The input is a Alpha scope sample!!")
                CCF_DIRECTIONS = {
                    0: "Posterior_to_anterior",
                    1: "Inferior_to_superior",
                    2: "Left_to_right",
                }
        swaps, flips = OrientationUtils.get_adjustments(metadata['axes'], CCF_DIRECTIONS)
        for a, b in swaps:
            soma_locations[:, [a, b]] = soma_locations[:, [b, a]]
        image_shape = input_img.shape
        for ax in flips:
            soma_locations[:, ax] = image_shape[ax] - 1 - soma_locations[:, ax]
        return soma_locations

    def resample_isotropic(self, soma_locations: np.ndarray, scale: list) -> np.ndarray:
        """
        Resample soma locations to isotropic resolution.

        Parameters
        ----------
        soma_locations : np.ndarray
            Soma locations to resample.
        scale : list
            Scaling factors for each axis.

        Returns
        -------
        np.ndarray
            Resampled soma locations.
        """
        scaled_cells = []
        for cell in soma_locations:
            scaled_cells.append(
                [cell[0] * scale[0], cell[1] * scale[1], cell[2] * scale[2]]
            )
        return np.array(scaled_cells)

    def soma_physical_to_index(self, soma_locations_xyz: np.ndarray, level: int) -> np.ndarray:
        """
        Convert soma locations from physical to voxel space.

        Parameters
        ----------
        soma_locations_xyz : np.ndarray
            Soma locations in physical coordinates.
        level : int
            Pyramid level for voxelization.

        Returns
        -------
        np.ndarray
            Voxel coordinates.
        """
        anisotropy = np.array([0.748, 0.748, 1.0])
        soma_locations = np.zeros((len(soma_locations_xyz), 3))
        for i, xyz in enumerate(soma_locations_xyz):
            soma_locations[i] = CoordinateConverter.to_voxels(xyz, anisotropy, level)
        return soma_locations 
    
    
    
    def apply_simpleitk_transform(self, transform_path: str, points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a SimpleITK transform or displacement field to points.
        
        Parameters
        ----------
        transform_path : str
            Path to the transform file (.nrrd, .tfm, etc.)
        points_df : pd.DataFrame
            DataFrame with points in physical coordinates (columns: x, y, z)
            
        Returns
        -------
        pd.DataFrame
            Transformed points in physical coordinates
        """
        try:
            # Check if it's a displacement field (image) or transform
            if transform_path.endswith('.nrrd') or transform_path.endswith('.nii') or transform_path.endswith('.nii.gz'):
                # Treat as displacement field (image)
                displacement_field = sitk.ReadImage(transform_path)
                
                # Convert points to SimpleITK format
                points_sitk = []
                for _, row in points_df.iterrows():
                    points_sitk.append([row['x'], row['y'], row['z']])
                
                # Apply displacement field
                transformed_points = []
                for point in points_sitk:
                    try:
                        # Convert point to SimpleITK format (tuple of doubles)
                        point_tuple = tuple(float(p) for p in point)
                        
                        # Convert physical point to index in displacement field
                        index = displacement_field.TransformPhysicalPointToIndex(point_tuple)
                        
                        # Check if index is within bounds
                        size = displacement_field.GetSize()
                        if (0 <= index[0] < size[0] and 
                            0 <= index[1] < size[1] and 
                            0 <= index[2] < size[2]):
                            
                            # Get displacement vector at this index
                            displacement_vector = displacement_field.GetPixel(index)
                            
                            # Apply displacement to original point
                            transformed_point = [point[i] + displacement_vector[i] for i in range(3)]
                            transformed_points.append(transformed_point)
                        else:
                            # Point outside displacement field bounds, keep original
                            print(f"Warning: Point {point} outside displacement field bounds")
                            transformed_points.append(point)
                            
                    except Exception as e:
                        print(f"Warning: Could not transform point {point}: {e}")
                        # Keep original point if transform fails
                        transformed_points.append(point)
                
                # Return as DataFrame
                return pd.DataFrame(transformed_points, columns=["x", "y", "z"])
                
            else:
                # Treat as transform file
                transform = sitk.ReadTransform(transform_path)
                
                # Convert points to SimpleITK format
                points_sitk = []
                for _, row in points_df.iterrows():
                    points_sitk.append([row['x'], row['y'], row['z']])
                
                # Apply transform
                transformed_points = []
                for point in points_sitk:
                    try:
                        transformed_point = transform.TransformPoint(point)
                        transformed_points.append(transformed_point)
                    except Exception as e:
                        print(f"Warning: Could not transform point {point}: {e}")
                        # Keep original point if transform fails
                        transformed_points.append(point)
                
                # Return as DataFrame
                return pd.DataFrame(transformed_points, columns=["x", "y", "z"])
            
        except Exception as e:
            print(f"Warning: Could not load or apply transform {transform_path}: {e}")
            # Return original points if transform fails
            return points_df 
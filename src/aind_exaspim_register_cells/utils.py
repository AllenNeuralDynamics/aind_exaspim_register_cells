"""Utilities for coordinate conversion and orientation handling.

This module provides classes for handling coordinate transformations between
different coordinate systems (physical, voxel, ANTs) and for managing
orientation adjustments of arrays and axes.
"""

import ast
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

class OrientationUtils:
    """
    Utilities for handling orientation, swaps, and flips of axes and arrays.
    """
    @staticmethod
    def get_adjustments(
        axes: List[Dict[str, Any]], orientation: Dict[int, str]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Computes the necessary swaps and flips to adjust the orientation of axes.

        Parameters
        ----------
        axes : list of dict
            List containing axis information from the acquisition.json file.
        orientation : dict
            Reference orientation for each axis dimension.

        Returns
        -------
        swaps : list of tuple
            Tuples of dimensions to be swapped.
        flips : list of int
            Axes that need to be flipped.
        """
        flips: List[int] = []
        swaps: List[Tuple[int, int]] = []
        for i in range(len(axes)):
            ax = axes[i]
            dim = ax["dimension"]
            direction = ax["direction"].lower()

            if orientation[dim].lower() == direction:
                continue

            for idx, d in orientation.items():
                if d.lower() == direction:
                    swaps.append((dim, idx))
                elif d.lower() == "_".join(direction.split("_")[::-1]):
                    swaps.append((dim, idx))
                    flips.append(idx)
        return swaps, flips

    @staticmethod
    def adjust_array(arr: np.ndarray, swaps: List[Tuple[int, int]], flips: List[int]) -> np.ndarray:
        """
        Adjusts a NumPy array by performing axis swaps and flips.

        Parameters
        ----------
        arr : np.ndarray
            The input NumPy array to be adjusted.
        swaps : list of tuple
            Tuples representing the axes to be swapped.
        flips : list of int
            Axes that should be flipped.

        Returns
        -------
        np.ndarray
            The adjusted NumPy array after the swaps and flips are applied.
        """
        if swaps:
            in_axis, out_axis = zip(*swaps)
            arr = np.moveaxis(arr, in_axis, out_axis)
        if flips:
            arr = np.flip(arr, axis=flips)
        return arr

    @staticmethod
    def get_orientation_transform(orientation_in: str, orientation_out: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes orientation acronyms (i.e. spr) and creates a conversion matrix for converting from one to another.

        Parameters
        ----------
        orientation_in : str
            The current orientation of image or cells (i.e. spr)
        orientation_out : str
            The orientation to convert to (i.e. ras)

        Returns
        -------
        tuple
            The location of the values in the identity matrix with values (original, swapped, transform_matrix)
        """
        reverse_dict = {"r": "l", "l": "r", "a": "p", "p": "a", "s": "i", "i": "s"}
        input_dict = {dim.lower(): c for c, dim in enumerate(orientation_in)}
        output_dict = {dim.lower(): c for c, dim in enumerate(orientation_out)}
        transform_matrix = np.zeros((3, 3))
        for k, v in input_dict.items():
            if k in output_dict.keys():
                transform_matrix[v, output_dict[k]] = 1
            else:
                k_reverse = reverse_dict[k]
                transform_matrix[v, output_dict[k_reverse]] = -1
        if orientation_in.lower() == "spl" or orientation_out.lower() == "spl":
            transform_matrix = abs(transform_matrix)
        original, swapped = np.where(transform_matrix.T)
        return original, swapped, transform_matrix

class CoordinateConverter:
    """
    Utilities for converting between physical, voxel, index, and ANTs spaces.
    """
    @staticmethod
    def to_voxels(xyz: np.ndarray, anisotropy: np.ndarray, level: int) -> np.ndarray:
        """
        Converts the given coordinate from physical to voxel space.

        Parameters
        ----------
        xyz : np.ndarray
            Physical coordinate to be converted to a voxel coordinate.
        anisotropy : np.ndarray
            Image to physical coordinates scaling factors.
        level : int
            Level in the zarr image pyramid.

        Returns
        -------
        np.ndarray
            Voxel coordinate of the input.
        """
        scaling_factor = 1.0 / 2**level
        voxel = scaling_factor * (xyz / anisotropy)
        # return np.round(voxel).astype(int)
        return voxel


    @staticmethod
    def load_soma_locations(somas_path: str, level: int) -> np.ndarray:
        """
        Reads text file containing soma locations stored in physical coordinates, then converts them to voxel coordinates.

        Parameters
        ----------
        somas_path : str
            Path to soma locations text file.
        level : int
            Level in the zarr image pyramid.

        Returns
        -------
        np.ndarray
            Voxel coordinates of the input.
        """
        with open(somas_path, "r") as f:
            soma_locations_txt = f.read().splitlines()
        anisotropy = np.array([0.748, 0.748, 1.0])  # HARD CODED FOR EXASPIM
        soma_locations = np.zeros((len(soma_locations_txt), 3))
        for i, xyz in enumerate(map(ast.literal_eval, soma_locations_txt)):
            soma_locations[i] = CoordinateConverter.to_voxels(xyz, anisotropy, level)
        return soma_locations
        
    @staticmethod
    def index_to_physical(ants_img: Any, index_batch: np.ndarray) -> np.ndarray:
        """
        Convert index coordinates to physical coordinates using ANTs image properties.

        Parameters
        ----------
        ants_img : Any
            ANTsImage or similar object with spacing, origin, direction.
        index_batch : np.ndarray
            Index coordinates.

        Returns
        -------
        np.ndarray
            Physical coordinates.
        """
        index_batch = np.asarray(index_batch)
        spacing = np.asarray(ants_img.spacing)
        origin = np.asarray(ants_img.origin)
        direction = np.asarray(ants_img.direction).reshape((3, 3))
        physical = origin + (index_batch * spacing) @ direction.T
        return physical

    @staticmethod
    def physical_to_index(ants_img: Any, physical_batch: np.ndarray) -> np.ndarray:
        """
        Convert physical coordinates to index coordinates using ANTs image properties.

        Parameters
        ----------
        ants_img : Any
            ANTsImage or similar object with spacing, origin, direction.
        physical_batch : np.ndarray
            Physical coordinates.

        Returns
        -------
        np.ndarray
            Index coordinates (rounded to int32).
        """
        physical_batch = np.asarray(physical_batch)
        spacing = np.asarray(ants_img.spacing)
        origin = np.asarray(ants_img.origin)
        direction = np.asarray(ants_img.direction).reshape((3, 3))
        relative = physical_batch - origin
        index = (relative @ np.linalg.inv(direction).T) / spacing
        # return np.round(index).astype(np.int32)
        return index


    @staticmethod
    def convert_to_ants_space(template: Any, cells: np.ndarray) -> np.ndarray:
        """
        Convert points from "index" space and places them into the physical space required for applying ants transforms for a given ANTsImage.

        Parameters
        ----------
        template : Any
            ANTsImage or similar object with orientation, dimension, spacing, origin, direction.
        cells : np.ndarray
            The location of cells in index space.

        Returns
        -------
        np.ndarray
            Points converted into ANTsPy physical space.
        """
        params = {
            "orientation": template.orientation,
            "dims": template.dimension,
            "scale": template.spacing,
            "origin": template.origin,
            "direction": template.direction[np.where(template.direction != 0)],
        }
        ants_pts = cells.copy()
        for dim in range(params["dims"]):
            ants_pts[:, dim] *= params["scale"][dim]
            ants_pts[:, dim] *= params["direction"][dim]
            ants_pts[:, dim] += params["origin"][dim]
        return ants_pts

    @staticmethod
    def convert_from_ants_space(template: Any, cells: np.ndarray) -> np.ndarray:
        """
        Convert points from the physical space of an ANTsImage and places them into the "index" space required for visualizing.

        Parameters
        ----------
        template : Any
            ANTsImage or similar object with orientation, dimension, spacing, origin, direction.
        cells : np.ndarray
            The location of cells in physical space.

        Returns
        -------
        np.ndarray
            Points converted from ANTsPy physical space to "index" space.
        """
        params = {
            "orientation": template.orientation,
            "dims": template.dimension,
            "scale": template.spacing,
            "origin": template.origin,
            "direction": template.direction[np.where(template.direction != 0)],
        }
        pts = cells.copy()
        for dim in range(params["dims"]):
            pts[:, dim] -= params["origin"][dim]
            pts[:, dim] *= params["direction"][dim]
            pts[:, dim] /= params["scale"][dim]
        return pts


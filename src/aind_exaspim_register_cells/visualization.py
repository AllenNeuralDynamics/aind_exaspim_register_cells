"""Image visualization utilities for ANTs images and overlays.

This module provides utilities for plotting and visualizing ANTs images,
including normalization, multi-slice plotting, and point overlays on
3D volumes.
"""

import matplotlib.pyplot as plt
import numpy as np
import ants

class ImageVisualizer:
    """
    Utilities for plotting and visualizing images and overlays.
    """
    @staticmethod
    def perc_normalization(
        ants_img: ants.ANTsImage, lower_perc: float = 2, upper_perc: float = 98
    ) -> ants.ANTsImage:
        """
        Percentile Normalization for ANTsImage.

        Parameters
        ----------
        ants_img : ants.ANTsImage
            The input image to normalize.
        lower_perc : float
            Lower percentile for normalization.
        upper_perc : float
            Upper percentile for normalization.

        Returns
        -------
        ants.ANTsImage
            Normalized image.
        """
        percentiles = [lower_perc, upper_perc]
        percentile_values = np.percentile(ants_img.view(), percentiles)
        assert percentile_values[1] > percentile_values[0]
        ants_img = (ants_img - percentile_values[0]) / (
            percentile_values[1] - percentile_values[0]
        )
        return ants_img

    @staticmethod
    def plot_antsimgs(ants_img: ants.ANTsImage, figpath: str = "", title: str = "", vmin: float = 0, vmax: float = 1.5, cmap: str = "gray") -> None:
        """
        Plot ANTs image in three orthogonal slices.

        Parameters
        ----------
        ants_img : ants.ANTsImage
            The image to plot.
        figpath : str
            Path to save the figure (optional).
        title : str
            Title of the plot.
        vmin : float
            Minimum color value.
        vmax : float
            Maximum color value.
        cmap : str
            Colormap.
        """
        ants_img = ants_img.numpy()
        half_size = np.array(ants_img.shape) // 2
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        ax[0].imshow(
            ants_img[half_size[0], :, :], cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax[1].imshow(
            ants_img[:, half_size[1], :], cmap=cmap, vmin=vmin, vmax=vmax
        )
        im = ax[2].imshow(
            ants_img[
                :,
                :,
                half_size[2],
            ],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.suptitle(title, y=0.9)
        plt.colorbar(
            im, ax=ax.ravel().tolist(), fraction=0.1, pad=0.025, shrink=0.7
        )
        plt.show()
        
        # plt.savefig(figpath, bbox_inches="tight", pad_inches=0.1)

    @staticmethod
    def soma_overlay_volumn(points: np.ndarray, volume: np.ndarray, title: str = None, figpath: str = None) -> None:
        """
        Overlay soma points on three orthogonal slices of a volume.

        Parameters
        ----------
        points : np.ndarray
            Soma coordinates.
        volume : np.ndarray
            3D image volume.
        title : str
            Title for the plot.
        figpath : str
            Path to save the figure (optional).
        """
        shape = volume.shape
        fig, axes = plt.subplots(1, 3, figsize=(11, 11))
        ImageVisualizer.overlay_slice(volume, points, axis='x', index=shape[0] // 2, ax=axes[0])
        ImageVisualizer.overlay_slice(volume, points, axis='y', index=shape[1] // 2, ax=axes[1])
        ImageVisualizer.overlay_slice(volume, points, axis='z', index=shape[2] // 2, ax=axes[2])
        fig.suptitle(title, fontsize=16, y=0.8)
        plt.tight_layout()
        plt.show()
        # plt.savefig(figpath, bbox_inches="tight", pad_inches=0.1)

    @staticmethod
    def overlay_slice(volume: np.ndarray, points: np.ndarray, axis: str = 'z', index: int = 25, ax=None) -> None:
        """
        Overlay points on a single slice of a 3D volume.

        Parameters
        ----------
        volume : np.ndarray
            3D image volume.
        points : np.ndarray
            Points to overlay.
        axis : str
            Axis to slice ('x', 'y', or 'z').
        index : int
            Index of the slice.
        ax : matplotlib.axes.Axes
            Axis to plot on.
        """
        if axis == 'x':
            slice_img = volume[index, :, :]
            coords = points[:, [1, 2]]
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 'y':
            slice_img = volume[:, index, :]
            coords = points[:, [0, 2]]
            xlabel, ylabel = 'X', 'Z'
        elif axis == 'z':
            slice_img = volume[:, :, index]
            coords = points[:, [0, 1]]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")
        ax.imshow(slice_img, cmap='gray', vmin=0, vmax=1.5)
        if coords.shape[0] > 0:
            ax.scatter(coords[:, 1], coords[:, 0], c='red', s=3, label='Soma')
        ax.set_title(f'{axis.upper()}-slice at index {index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.axis('on') 
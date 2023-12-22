import open3d as o3d
import numpy as np
import torch

class GridOccupancyMap3D:
    """
    This class implements a 3D grid map using numpy arrays.
    """

    def __init__(
            self, min_bound: np.ndarray,
            max_bound: np.ndarray, cell_size: float = 0.01) -> None:
        """
        Initialize the grid map.

        Args:
            min_bound (np.ndarray): 
            minimum x, y, z coordinates of the bounding box
            max_bound (np.ndarray): 
            maximum x, y, z coordinates of the bounding box
            cell_size (float, optional): 
            size of the grid cells. Defaults to 0.01.
        """
        # Calculate the size of the bounding box
        # self.cpu_count = multiprocessing.cpu_count()
        # print("Number of cpus: ", self.cpu_count)
        self.cell_size = cell_size
        container_x_size = max_bound[0] - min_bound[0]
        container_y_size = max_bound[1] - min_bound[1]
        self.max_height = max_bound[2]
        print("container_x_size: ", container_x_size)
        print("container_y_size: ", container_y_size)
        print("max_height: ", self.max_height)
        self.container_volume = \
            container_x_size * container_y_size * max_bound[2]
        print("container_volume: ", self.container_volume)
        # Calculate the number of grid cells needed to fill the bounding box area
        num_x_cells = int(np.ceil(container_x_size / self.cell_size))
        num_y_cells = int(np.ceil(container_y_size / self.cell_size))

        # Calculate the size of each grid cell to fill the bounding box area
        new_cell_size_x = container_x_size / num_x_cells
        new_cell_size_y = container_y_size / num_y_cells

        # Update the grid parameters and reset the grid
        self.x_size = num_x_cells
        self.y_size = num_y_cells
        # Adjust cell size based on both dimensions
        self.cell_size = max(new_cell_size_x, new_cell_size_y)
        self.grid = np.zeros((self.x_size, self.y_size), dtype=np.float16)

    def generate_grid_boxes(
            self) -> list[o3d.geometry.AxisAlignedBoundingBox]:
        """
        Generate a list of AxisAlignedBoundingBoxes to visualize the grid.

        Returns:
            list[o3d.geometry.AxisAlignedBoundingBox]: 
            list of AxisAlignedBoundingBoxes
        """
        grid_boxes = []

        grid_x, grid_y = np.meshgrid(
            np.arange(self.x_size),
            np.arange(self.y_size), indexing='ij'
        )
        max_heights = self.grid[grid_x, grid_y]

        # Set the minimum height to 0.001
        max_heights = np.where(max_heights == 0, 0.001, max_heights)
        min_bounds = np.array([
            grid_x * self.cell_size,
            grid_y * self.cell_size,
            np.zeros_like(max_heights)
        ]).T
        max_bounds = np.array([
            (grid_x + 1) * self.cell_size,
            (grid_y + 1) * self.cell_size,
            max_heights
        ]).T

        for min_bound, max_bound in zip(
            min_bounds.reshape(-1, 3),
            max_bounds.reshape(-1, 3),
        ):
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            box.color = [1, 0, 0]
            grid_boxes.append(box)

        return grid_boxes

    def xyz_to_2Dgrid(
            self, xyz: np.ndarray,
            cell_size: float = 0.01) -> None:
        """
        Downsample a 2D point cloud by taking the first point 
        in each cell of a grid.

        Args:
            xyz (ndarray): xyz coordinates of the point cloud
            cell_size (float): size of the grid cells

        Returns:
            None
        """
        # Round the x and y coordinates to the nearest cell_size
        xyz_rounded = np.round(xyz[:, :2] / cell_size) * cell_size
        xyz_grid = np.hstack((xyz_rounded, xyz[:, 2][:, np.newaxis]))

        # Find unique indices based on rounded coordinates
        unique_indices = np.unique(
            xyz_rounded, axis=0, return_index=True
        )[1]

        # Select representative points based on unique indices
        xyz_grid = xyz_grid[unique_indices]

        # Convert x and y coordinates to grid indices
        x_idx = np.floor(xyz[:, 0] / cell_size).astype(np.uint32)
        y_idx = np.floor(xyz[:, 1] / cell_size).astype(np.uint32)

        # Reset grid
        self.grid = np.zeros((self.x_size, self.y_size), dtype=np.float16)

        # Masking to efficiently assign z-coordinates to grid cells
        self.grid[x_idx, y_idx] = xyz[:, 2]

    def get_fill_rate(self) -> float:
        """
        This function calculates the volume of the grid 
        that is filled with points and returns the fill rate.

        Returns:
            float: fill rate of the grid
        """
        # Calculate the volume of the filled grid
        filled_volume = np.sum(self.grid.flatten()) * self.cell_size ** 3
        # Calculate the fill rate
        fill_rate = (filled_volume / self.container_volume)*100

        return fill_rate

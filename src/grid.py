import open3d as o3d
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import cv2
import scipy

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
        # self.loaded_grid = np.load("reference_grid.npy")
        self.action = False
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
        self.cell_status = np.zeros((self.x_size, self.y_size), dtype=np.uint8)
        self.grid = np.zeros((self.x_size, self.y_size), dtype=np.float16)
        self.reference_grid = np.zeros((self.x_size, self.y_size), dtype=np.float16)
        self.dz = np.zeros((self.x_size, self.y_size), dtype=np.float16)
        self.grid_snapshots = deque(maxlen=2)
        self.dz_snapshots = deque(maxlen=2)
        self.laplacian_kernel = torch.tensor([[[
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]]], dtype=torch.float32)
        self.gaussian_kernel = torch.tensor([[[
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]]], dtype=torch.float32)/16
        self.action = False
        cv2.namedWindow('Convolution Result Heatmap',
                        cv2.WINDOW_NORMAL)  # Create a named window
        cv2.resizeWindow('Convolution Result Heatmap', 800,
                         600)  # Set initial width and height

        # print("kernel shape: ", self.laplacian_kernel.shape)
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
        max_heights = self.reference_grid[grid_x, grid_y]
        # print("max_heights: ", np.max(max_heights))
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

        for min_bound, max_bound, cell_status in zip(
            min_bounds.reshape(-1, 3),
            max_bounds.reshape(-1, 3),
            self.cell_status.T.reshape(-1, 1)
        ):
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            if cell_status == 0:
                box.color = [1, 0, 0]
            else:
                box.color = [0, 1, 0]
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
        # if len(self.grid_snapshots) == 0:
        #     self.first_grid = self.grid.copy()
        self.grid_snapshots.append(self.grid.copy())


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

    def remove_outliers(self, dz: np.ndarray, threshold: float = 2.0,
                        window_size: int = 7) -> np.ndarray:
        # Create a padded array to handle boundary cases
        p = (window_size // 2) + 2  # Padding size
        padded_dz = np.pad(dz, ((p, p), (p, p)),
                           mode='constant', constant_values=np.nan)

        # Create a rolling window of shape (5, 5)
        rolling_window = np.lib.stride_tricks.sliding_window_view(
            padded_dz, (window_size, window_size)
        )

        # Compute the mean of the rolling window along the last two axes
        local_means = np.nanmedian(rolling_window, axis=(-2, -1))

        # Compute the absolute deviation of each element from the local mean
        abs_deviation = np.abs(dz - local_means[2:-2, 2:-2])

        # Identify outliers based on the threshold
        outliers = abs_deviation > threshold * \
            np.nanstd(rolling_window, axis=(-2, -1))[2:-2, 2:-2]

        # Replace outliers with the local mean
        central_value = np.where(outliers, local_means[2:-2, 2:-2], dz)

        return central_value

    def get_grid_dz(self, dz_threshold:float=0.1) -> np.ndarray:
        """
        This function calculates the height difference between
        the last two grid snapshots.
        """
        dz = self.grid_snapshots[-1] - self.reference_grid
        dz_action = self.grid_snapshots[-1] - self.grid_snapshots[-2]

        dz = self.remove_outliers(dz, threshold=1)
        dz_action = self.remove_outliers(dz_action, threshold=1)
        dz = dz.astype(np.float32)
        dz_action = dz_action.astype(np.float32)

        # detect if a cell is static or dynamic based on the height difference dz
        # Calculate mean height differences within a moving window
        mean_dz = np.abs(
            scipy.ndimage.uniform_filter(dz, size=3)
        )
        # Identify dynamic cells based on the threshold
        self.cell_status = np.where(mean_dz > dz_threshold, 1, 0)
        # print("mean_dz: ", np.max(mean_dz))
        self.generate_dynamic_clusters_masks()
        return dz, dz_action

    def grid_dz_conv_2d(self, visualize=False) -> np.ndarray:
        """
        This function calculates the height difference between
        the last two grid snapshots. It uses a 2D convolution
        and shows it with opencv as 2d heatmap.
        """
        if len(self.grid_snapshots) == 1:
            self.fixed_snapshot = self.grid_snapshots[0].copy()
        if len(self.grid_snapshots) > 1:
            dz, dz_action = self.get_grid_dz()
            self.detect_action(dz_action, dz)
            # dz = self.cell_status
            dz = torch.from_numpy(dz)
            dz = dz.unsqueeze(0).unsqueeze(0)
            # # print("dz shape: ", dz.shape)
            dz = dz.to(torch.float32)
            dz = torch.nn.functional.conv2d(dz, self.gaussian_kernel, padding=1)
            # # print("dz shape after conv2d: ", dz.shape)

            # # dz = torch.nn.functional.avg_pool2d(dz, 2, stride=2, padding=1)
            # # print("dz shape after pool2d: ", dz.shape)

            # # dz = torch.nn.functional.conv2d(dz, self.laplacian_kernel, padding=1)
            dz = dz.squeeze(0).squeeze(0)
            dz = dz.numpy()
            if visualize:
                self.visualize_convolution_heatmap(dz)
            return dz
        return None


    def generate_dynamic_clusters_masks(self) -> np.ndarray:
        """
        Generates masks for dynamic cell clusters.

        Args:
            threshold (float): Threshold defining dynamic cells.
            window_size (int): Size of the moving window.

        Returns:
            np.ndarray: Masks representing dynamic cell clusters.
        """
        # Label connected components in the dynamic_cells array
        labeled, num_features = scipy.ndimage.label(self.cell_status)
        print("num_features: ", num_features)
        # Generate masks for dynamic cell clusters
        masks = []
        for label in range(1, num_features + 1):
            mask = labeled == label
            masks.append(mask)

        self.dynamic_cluster = np.array(masks)

    def detect_action(self, dz_action, dz, threshold_action: float = 0.1, threshold_stable: float = 0.03) -> bool:
        """
        Detects if there's an action in the grid based on height variations between frames.

        Args:
            threshold_action (float): Threshold defining an action.
            threshold_stable (float): Threshold defining stability.

        Returns:
            bool: True if an action is detected, False if stable.
        """
        # if len(self.grid_snapshots) < 2:
        #     # Not enough frames to compare yet
        #     self.action = False

        # Calculate the absolute deviation of each element from the mean
        abs_deviation = np.abs(dz_action)
        
        # Calculate the maximum deviation in the grid
        max_deviation = np.max(abs_deviation)
        neg_dz = np.where(dz < 0, True, False)
        min_deviation = np.mean(dz[neg_dz])
        # Determine if the grid is stable or if an action is detected
        if max_deviation > threshold_action:
            self.action = True  # Action detected
            if min_deviation < -0.1:
                self.reference_grid = self.grid
                # print("lowest grid updated dz negative ", min_deviation)
        elif max_deviation < threshold_stable:
            if self.action:
                self.reference_grid = self.grid_snapshots[-1].copy()
                # np.save("reference_grid.npy", self.reference_grid)
                # print("lowest grid updated")
            self.action = False



    def grid_neighbors_dz(self, visualize=False) -> np.ndarray:
        """
        This function calculates the height difference between 
        each grid cell and its neighboring cells.

        Returns:
            np.ndarray: height difference between each grid cell 
            and its neighboring cells
        """
        # Calculate the height difference between each grid cell and its neighbors
        grid_dz = np.zeros((self.x_size, self.y_size))
        grid_dz[:-1, :] = self.grid[1:, :] - self.grid[:-1, :]
        grid_dz[1:, :] = self.grid[:-1, :] - self.grid[1:, :]
        grid_dz[:, :-1] = self.grid[:, 1:] - self.grid[:, :-1]
        grid_dz[:, 1:] = self.grid[:, :-1] - self.grid[:, 1:]
        if visualize:
            self.visualize_convolution_heatmap(grid_dz)
        return grid_dz

    def visualize_convolution_heatmap(self, conv_result: np.ndarray):
        """
        Visualize the convolution result as a heatmap using matplotlib.

        Args:
            conv_result (np.ndarray): Convolution result to visualize
        """
        # print(conv_result.shape, " -- ", conv_result.dtype)
        conv_result = conv_result.astype(np.float32)
        conv_result_normalized = cv2.normalize(
            conv_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        heatmap = cv2.applyColorMap(conv_result_normalized, cv2.COLORMAP_HOT)

        cv2.imshow('Convolution Result Heatmap', heatmap)
        key = cv2.waitKey(1)  # Update visualization (1 ms delay)
        if key == ord('q'):  # Press 'q' to close the window
            cv2.destroyAllWindows()


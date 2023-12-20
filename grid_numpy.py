import open3d as o3d
import numpy as np

class GridOccupancyMap3D:
    def __init__(self, min_bound, max_bound, cell_size):
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
        self.container_volume = container_x_size * container_y_size * max_bound[2]
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
        self.cell_size = max(new_cell_size_x, new_cell_size_y)  # Adjust cell size based on both dimensions
        self.grid = np.zeros((self.x_size, self.y_size), dtype=np.float16)
        # np.zeros((self.x_size, self.y_size))

        def is_occupied(self, x, y, z):
            grid_x = int(x / self.cell_size)
            grid_y = int(y / self.cell_size)

            if 0 <= grid_x < self.x_size and 0 <= grid_y < self.y_size:
                return self.grid[grid_x, grid_y]
            else:
                print("Out of grid bounds")
                return False

    def simulate_square_pointcloud(self, x, y, z, size):
        # this function takes the x, y, z coordinates of the center of a square
        # and the size of the square and returns a point cloud of the square with
        # random points all along the square area

        # create random points inside the square area and add random noise.
        x_points = np.random.uniform(x - size/2, x + size/2, 1000)
        y_points = np.random.uniform(y - size/2, y + size/2, 1000)
        z_points = np.random.uniform(z-size/100, z, 1000)

        xyz = np.array([x_points, y_points, z_points]).T
        xyz += np.random.normal(0, 0.01, xyz.shape)

        return xyz
    def generate_grid_boxes(self):
        grid_boxes = []

        grid_x, grid_y = np.meshgrid(
            np.arange(self.x_size), 
            np.arange(self.y_size), indexing='ij'
            )
        max_heights = self.grid[grid_x, grid_y]

        # mean, std = cell.get_z_variation_stats()  # Update this line as needed
        # dynamic = std > 1.4

        max_heights = np.where(max_heights == 0, 0.001, max_heights)
        min_bounds = np.array([grid_x * self.cell_size, grid_y * self.cell_size, np.zeros_like(max_heights)]).T
        max_bounds = np.array([(grid_x + 1) * self.cell_size, (grid_y + 1) * self.cell_size, max_heights]).T

        # boxes = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        # colors = np.where(dynamic, [1, 0, 0], [0, 0, 0])

        for min_bound, max_bound in zip(
                min_bounds.reshape(-1, 3), 
                max_bounds.reshape(-1, 3), 
                # colors.reshape(-1, 3)
            ):
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            box.color = [1, 0, 0]
            grid_boxes.append(box)
        
        return grid_boxes

    def update_grid_cells(self, xyz):
        # Create coordinate grids for x and y
        x_coords = np.arange(0, (self.x_size + 1) * self.cell_size, self.cell_size)
        y_coords = np.arange(0, (self.y_size + 1) * self.cell_size, self.cell_size)

        # Create masks for x and y coordinates
        # mask_z = xyz[:, 2] <= self.max_height
        mask_x = np.logical_and(x_coords[:-1, None] <= xyz[:, 0], xyz[:, 0] < x_coords[1:, None])
        mask_y = np.logical_and(y_coords[:-1, None] <= xyz[:, 1], xyz[:, 1] < y_coords[1:, None])
        # mask_x = np.logical_and(mask_x, mask_z)
        # mask_y = np.logical_and(mask_y, mask_z)
        grid_x = np.argmax(mask_x, axis=0) - 1
        grid_y = np.argmax(mask_y, axis=0) - 1

        # Filter points for each cell using masks
        grid_points = np.zeros((self.x_size, self.y_size, xyz.shape[1]))
        np.add.at(grid_points, (grid_x, grid_y), xyz)
        # print(grid_points[10, 10, :])
        # Calculate maximum height for each cell
        max_height = np.max(grid_points, axis=2)
        # filter out points that are under the max height
        max_height = np.where(max_height <= self.max_height, self.max_height, 0)
        # Update the grid
        self.grid = max_height

    def xyz_to_2Dgrid(
            self, xyz: np.ndarray,
            cell_size: float = 0.01):
        """
        Downsample a 2D point cloud by taking the first point in each cell of a grid.

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
        unique_indices = np.unique(xyz_rounded, axis=0, return_index=True)[1]

        # Select representative points based on unique indices
        xyz_grid = xyz_grid[unique_indices]

        # Convert x and y coordinates to grid indices
        x_idx = np.floor(xyz[:, 0] / cell_size).astype(np.uint32)
        y_idx = np.floor(xyz[:, 1] / cell_size).astype(np.uint32)

        # Reset grid
        self.grid = np.zeros((self.x_size, self.y_size), dtype=np.float16)

        # Masking to efficiently assign z-coordinates to grid cells
        self.grid[x_idx, y_idx] = xyz[:, 2]

    def get_fill_rate(self):
        """
        This function calculates the volume of the grid 
        that is filled with points and returns the fill rate.
        """
        # Calculate the volume of the filled grid
        filled_volume = np.sum(self.grid.flatten()) * self.cell_size ** 3
        # Calculate the fill rate
        fill_rate = (filled_volume / self.container_volume)*100

        return fill_rate
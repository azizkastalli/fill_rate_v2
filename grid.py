import open3d as o3d
import numpy as np
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from time import time, sleep
from cell import cell

class GridOccupancyMap3D:
    def __init__(self, min_bound, max_bound, cell_size):
        # Calculate the size of the bounding box
        # self.cpu_count = multiprocessing.cpu_count()
        # print("Number of cpus: ", self.cpu_count)
        self.cell_size = cell_size
        bb_x_size = max_bound[0] - min_bound[0]
        bb_y_size = max_bound[1] - min_bound[1]

        # Calculate the number of grid cells needed to fill the bounding box area
        num_x_cells = int(np.ceil(bb_x_size / self.cell_size))
        num_y_cells = int(np.ceil(bb_y_size / self.cell_size))

        # Calculate the size of each grid cell to fill the bounding box area
        new_cell_size_x = bb_x_size / num_x_cells
        new_cell_size_y = bb_y_size / num_y_cells

        # Update the grid parameters and reset the grid
        self.x_size = num_x_cells
        self.y_size = num_y_cells
        self.cell_size = max(new_cell_size_x, new_cell_size_y)  # Adjust cell size based on both dimensions
        self.grid = dict( ((x, y), cell()) for x in range(self.x_size) for y in range(self.y_size))
        # np.zeros((self.x_size, self.y_size))

    def is_occupied(self, x, y, z):
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)

        if 0 <= grid_x < self.x_size and 0 <= grid_y < self.y_size:
            return self.grid[grid_x][grid_y]
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
        for (grid_x, grid_y), cell in self.grid.items():
            max_height = cell.get_z()
            mean, std = cell.get_z_variation_stats()
            dynamic = False
            if std > 1.4:
                dynamic = True
                # print("mean: ", mean, "std: ", std)

            max_height = 0.001 if max_height == 0 else max_height
            min_bound = [grid_x * self.cell_size, grid_y * self.cell_size, 0]
            max_bound = [
                (grid_x + 1) * self.cell_size, 
                (grid_y + 1) * self.cell_size, 
                max_height
            ]
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            if dynamic:
                box.color = [1, 0, 0]
            else:
                box.color = [0, 0, 0]
            grid_boxes.append(box)
        return grid_boxes


    def process_grid_cell(self, grid_xy, cell_size, xyz):
        # Define grid boundaries
        # sleep(0.01)
        grid_x, grid_y = grid_xy
        min_x = grid_x * cell_size
        max_x = (grid_x + 1) * cell_size
        min_y = grid_y * cell_size
        max_y = (grid_y + 1) * cell_size

        # Filter points that fall within the current grid cell using NumPy indexing
        mask_x = np.logical_and(min_x <= xyz[:, 0], xyz[:, 0] < max_x)
        mask_y = np.logical_and(min_y <= xyz[:, 1], xyz[:, 1] < max_y)
        points_in_cell_mask = np.logical_and(mask_x, mask_y)
        points_in_cell = xyz[points_in_cell_mask]


        # Find maximum height among points in this cell
        max_height = np.max(points_in_cell[:, 2]) if len(points_in_cell) > 0 else 0

        return grid_x, grid_y, max_height


    def update_grid_cells(self, xyz):
        # Create a pool of processes
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Define partial function for the process_grid_cell function
            partial_process = partial(self.process_grid_cell, cell_size=self.cell_size, xyz=xyz)

            # Generate grid coordinates to map the function over
            grid_coordinates = [(grid_x, grid_y) for grid_x in range(self.x_size) for grid_y in range(self.y_size)]

            # Map the function over grid coordinates using multiprocessing
            results = [result for result in executor.map(partial_process, grid_coordinates)]

        # Update the grid with the results
        for grid_x, grid_y, max_height in results:
            self.grid[(grid_x, grid_y)].update_z_variation(max_height)
            
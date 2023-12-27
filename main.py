from time import time
import numpy as np
import open3d as o3d
from src.grid import GridOccupancyMap3D
from src.capture import Capture
import sys
from src.utils import segment_flats


if __name__ == '__main__':
    # create a coordinate frame
    coordinate_frame = o3d.geometry\
                          .TriangleMesh\
                          .create_coordinate_frame(size=0.3)
    # create an axis aligned bounding box from min max bounds
    min_bound = np.array([0, 0, 0])
    # 0.33, 0.478, 0.47
    # T = [0.1, 0.098, 1.42]
    T = [0.5, 0, 1.37]
    max_bound = np.array([ 0.7, 0.7, 1])
    # create the bounding box
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bounding_box.color = [0, 0, 0]  # Set color to red
    cell_size = 0.01

    # create a grid map
    grid_map = GridOccupancyMap3D(min_bound, max_bound, cell_size=cell_size)
    visualization = True
    show_boxes = True

    # connect kinect
    cap = Capture()
    if visualization:
        vis = o3d.visualization.Visualizer()
        vis.create_window(  
            width=1280, 
            height=720, 
            left=0, 
            top=0, 
            visible=True
        )
        vis.add_geometry(coordinate_frame)

    kinect_pcd = o3d.geometry.PointCloud()
    fps_timer = time()
    frame_count = 0
    fps = 0
    while True:
        start = time()
        pcd = cap.get_raw_pcd()
        # pcd = segment_flats(pcd)
        pcd.transform(np.array([[1, 0, 0, T[0]],
                                [0, 1, 0, T[1]],
                                [0, 0, 1, T[2]],
                                [0, 0, 0, 1]]))
        pcd = pcd.crop(bounding_box)
        xyz = np.asarray(pcd.points)
        color = np.asarray(pcd.colors)
        kinect_pcd.points = o3d.utility.Vector3dVector(xyz)
        kinect_pcd.colors = o3d.utility.Vector3dVector(color)
        grid_map.xyz_to_2Dgrid(xyz)
        if show_boxes:
            boxes = grid_map.generate_grid_boxes()
        fill_rate = grid_map.get_fill_rate()
        grid_dz = grid_map.grid_dz_conv_2d(visualize=True)
        stop = time()
        # Print the fill rate
        sys.stdout.flush()
        # Calculate FPS
        frame_count += 1
        if time() - fps_timer > 1:  # Calculate FPS every second
            fps = frame_count / (time() - fps_timer)
            frame_count = 0
            fps_timer = time()

        execution_time = (stop-start)*1000
        sys.stdout.write(
            f'  Fill rate: {fill_rate:.2f}%  || Time exeuction: {execution_time:.2f} ms || FPS: {fps:.2f} || Action: {np.mean(grid_map.action)} \r')
        if visualization:
            vis.add_geometry(kinect_pcd, reset_bounding_box=False)
            vis.add_geometry(bounding_box, reset_bounding_box=False)
            vis.add_geometry(coordinate_frame, reset_bounding_box=False)
            if show_boxes:
                [vis.add_geometry(box, reset_bounding_box=False)
                  for box in boxes]
            vis.poll_events()
            vis.update_renderer()
            vis.clear_geometries()

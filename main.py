from time import time
import numpy as np
import open3d as o3d
from grid_numpy import GridOccupancyMap3D
# from grid import GridOccupancyMap3D
from capture import Capture
import sys



if __name__ == '__main__':
    # create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    # create an axis aligned bounding box from min max bounds
    min_bound = np.array([0, 0, 0])
    # 0.33, 0.478, 0.41
    max_bound = np.array([0.33, 0.478, 0.475])
    # create the bounding box
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bounding_box.color = [0, 0, 0]  # Set color to red
    cell_size = 0.01

    # create a grid map
    grid_map = GridOccupancyMap3D(min_bound, max_bound, cell_size=cell_size)

    # pcd_fixed = o3d.geometry.PointCloud()
    # xyz_1 = grid_map.simulate_square_pointcloud(3, 3.5, 1, 1)
    # xyz_2 = grid_map.simulate_square_pointcloud(7, 3.5, 2, 1)
    # xyz_fixed = np.concatenate((xyz_1, xyz_2))
    # pcd_fixed.points = o3d.utility.Vector3dVector(xyz_fixed)
    # pcd_fixed.paint_uniform_color([0, 0, 0])
    visualization = False


    # connect kinect
    cap = Capture()
    if visualization:
        vis = o3d.visualization.Visualizer()
        vis.create_window(  width=1280, 
                        height=720, 
                        left=0, 
                        top=0, 
                        visible=True
                    )
        vis.add_geometry(coordinate_frame)

    first = True
    kinect_pcd = o3d.geometry.PointCloud()
    fps_timer = time()
    frame_count = 0
    fps = 0
    show_boxes = False
    while True:
        start = time()
        pcd = cap.get_raw_pcd()
        #pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd.transform(np.array([[1, 0, 0, 0.1],
                                [0, 1, 0, 0.098],
                                [0, 0, 1, 1.38],
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
            f'  Fill rate: {fill_rate:.2f}%  || Time exeuction: {execution_time:.2f} ms || FPS: {fps:.2f}  \r')
        # if first:
        if visualization:
            vis.add_geometry(kinect_pcd, reset_bounding_box=False)
            # vis.add_geometry(pcd_fixed)
            # vis.add_geometry(coordinate_frame)
            vis.add_geometry(bounding_box, reset_bounding_box=False)
            vis.add_geometry(coordinate_frame, reset_bounding_box=False)
            # first = False
        # vis.update_geometry(kinect_pcd)
        # vis.update_geometry(pcd_fixed)
        # vis.update_geometry(coordinate_frame)
        # vis.update_geometry(bounding_box)
            if show_boxes:
                [vis.add_geometry(box, reset_bounding_box=False) for box in boxes]
        # vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
            vis.clear_geometries()

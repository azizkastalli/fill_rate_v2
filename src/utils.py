import numpy as np
import open3d as o3d


def segment_flats(
        pcd:o3d.geometry.PointCloud, 
        flat_thresh:float=.8) -> o3d.geometry.PointCloud:
    """
    Extract 'flats' (horizontal elements) from the point 
    cloud based on the vertical orientation of the surface normal
    :param pc: pcd to segment
    :param wall_thresh: selection threshold for z co-ordinate of surface normal
    :return: points of horz elements of pc
    """
    if not np.asarray(pcd.normals).size:  # If normals aren't available yet
        pcd.estimate_normals()  # Estimating normals

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # threshld for a point to count as a flat, if z normal crd 
    # is in [thresh, MAX] or [MIN, -thresh]
    pts_up = pts[normals[:, -1] > 1 - flat_thresh]
    # pts = np.asarray(self.pcd.points)
    pts_down = pts[normals[:, -1] < -flat_thresh]

    flats = o3d.geometry.PointCloud()
    flats.points = o3d.utility.Vector3dVector(
        np.concatenate([pts_up, pts_down])
        )
    return flats
import open3d as o3d
import numpy as np

class Capture():
    def __init__(self, instrinsic_path='data/intrinsic.json'):
        """
        Initializes the capture class
        """
        self.intrinsic_path = instrinsic_path
        self.sensor = self._connect_sensor()        

    # set the sensors 
    def _connect_sensor(self):
        sensor = o3d.io.AzureKinectSensor(o3d.io.AzureKinectSensorConfig())
        if not sensor.connect(0):
            raise RuntimeError('Failed to connect to sensor')
        return sensor

    # get point clouds from each sensor
    def get_raw_pcd(self):
        """
        Captures from registered Kinect devices, pushes to queue
        :param device: device ID to capture from
        :param align_to_rgb: if True, aligns the depth image fisheye/barrel distortion to the flat RGB capture
        :param one_run: if True, runs the loop only once
        """
        azure_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            self.intrinsic_path
        )
        while True:
            capture = self.sensor.capture_frame(
                enable_align_depth_to_color=True
                )
            if capture is not None:
                rgbd = o3d.geometry\
                          .RGBDImage\
                          .create_from_color_and_depth(
                            color=capture.color,
                            depth=capture.depth,
                            convert_rgb_to_intensity=False
                        )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd, azure_intrinsic
                    )
                # _, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                #                                     std_ratio=1.4)
                # pcd = pcd.select_by_index(ind)
                break
        tf = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        pcd.transform(tf)
        return pcd
    

    # get rgb image from a sensor
    def get_rgb_image(self, index):
        capture = self.sensors[index].capture_frame(
                enable_align_depth_to_color=True
            )
        return np.asarray(capture.color)


if __name__ == '__main__':
    from capture import Capture

    cap = Capture()

    i = 0 
    while True:
        pcd = cap.get_raw_pcd()
        i += 1
        if i%100==0:
            o3d.visualization.draw_geometries([pcd])

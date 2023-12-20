import open3d as o3d
import numpy as np
from collections import deque



class cell:
    def __init__(self, max_z_size=3, ) -> None:
        self.__z_variation = deque(maxlen=max_z_size)
        self.neibhours = []

    def update_z_variation(self, z:int)-> None:
        self.__z_variation.append(z)

    def get_z_variation_stats(self,)-> tuple:
        return np.mean(self.__z_variation), np.std(self.__z_variation)

    # getters
    def get_z(self)-> int:
        return self.__z_variation[-1]
    def get_z_variation(self)-> deque:
        return self.__z_variation
    

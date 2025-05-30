
from pointCloudData import PointCloudData

#can you work to build the superquadric directly from the RGB-D image and mask removing the need for a point cloud object

class Superquadric:
    def __init__(self, object_ID, class_name, input_type, raw_data_1, raw_depth=None, raw_mask=None, camera_info=None):
        self.object_ID = object_ID
        self.class_name = class_name
        
        self.pcd = PointCloudData(object_ID, input_type, raw_data_1, raw_depth, raw_mask, camera_info)

        self.raw_data_1 = raw_data_1[0] #rgb raw, rgbd or pcd
        self.raw_depth = raw_depth[0]
        self.raw_mask = raw_mask[0]
        self.camera_info = camera_info

        self.modelParameters = self.defineParameters()
        self.model = self.createSuperquadric()

        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)

    def defineParameters(self):
        pass
    
    def createSuperquadric(self):
        #build the superquadric based on given parameters
        pass

    def getSuperquadric(self):
        return self.model
    
    def getPCD(self):
        return self.pcd

    def updateSuperquadric(self):
        #update superquadric: for future development
        pass
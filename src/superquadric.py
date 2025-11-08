#!/usr/bin/env python3.8

"""
    superquadric: estimates and firts superquadrics to partial point cloud model, 
                  shape values e1, e2 are estimated based on kurtosis of point cloud
                  size values a1, a2, a3 are estimated based on bounding box of point cloud
                  pose is estimated based on oriented bounding box of point cloud

    Input:
        pcd: point cloud of the object to fit superquadric to
        debug: if True, print debug information
    
    Output:
        superquadric: point cloud of the fitted superquadric
        pose: 3x3 rotation matrix of the superquadric  

    Developed by: Jayden Kanbour as part of undergraduate thesis for UNSW Computer Science and Engineering
    Date: 26th November 2025
    Email: jkanbour1@gmail.com
    UNSW Student Id: z5316799

"""

import numpy as np
import open3d as o3d
from scipy.stats import kurtosis
import time

class Superquadric:
    def __init__(self, pcd, downsample=30, debug=False):

        init_time = time.time()
        self.debug = debug
        self.downsample = downsample
        self.extent = None
        self.pose = None
        self.center = None

        # estimate e1, e2 for superquadric fitting
        self.e1, self.e2 = self.estimateE(pcd)

        # create superquadric mesh and pose
        self.superquadric = self.createSuperquadric(pcd, self.e1, self.e2)

        if self.debug:
            print(f"Superquadric: time: {time.time() - init_time:.3f}s")
            print(f"Superquadric: SQ time: {time.time() - init_time:.3f}s")
            print(f"Superquadric: Num Points: {len(self.superquadric.points)}")


    def estimateE(self, pcd):

        """
            Estimate superquadric shape parameters e1 and e2 based on point cloud kurtosis.
            
            e1 controls shape along z-axis (elongation/flattening)
            e2 controls shape in xy-plane (roundness/squareness)
        """

        try:
            points = np.asarray(pcd.points)
            center = pcd.get_center()
            centered_points = points - center

            # Compute Fisher kurtosis for x, y, z axes of the point cloud
            krt = kurtosis(centered_points, axis=0, fisher=True, bias=False)

            # Shape parameter along z-axis based on kurtosis (controls superquadric elongation or flattening)
            e1 = np.clip(1 + (krt[2] - 3) * 0.1, 0.3, 2.0)

            # Shape parameter along xy-plane based on average x and y kurtosis
            e2 = np.clip(1 + ((krt[0] + krt[1]) / 2 - 3) * 0.1, 0.3, 2.0)

            if self.debug:
                print(f"Estimated e1: {e1:.3f}, e2: {e2:.3f}")

            return e1, e2
        
        except Exception as e:
            print(f"superquadric [estimateE] Error: {e}")
            return 1.0, 1.0

    
    def createSuperquadric(self, pcd, e1, e2):
        """
            Create a superquadric point cloud fitted to the input point cloud.
            Uses oriented bounding box of the point cloud to determine size and pose.
        """

        try:
            # Check if point cloud is too flat
            points = np.asarray(pcd.points)
            ranges = points.max(axis=0) - points.min(axis=0)
            
            if self.debug:
                print(f"Point cloud ranges: X={ranges[0]:.4f}, Y={ranges[1]:.4f}, Z={ranges[2]:.4f}")
            
            # If any dimension is too small, add artificial thickness
            min_range = 0.01  # 1cm
            if np.any(ranges < min_range):
                flat_dim = np.argmin(ranges)
                if self.debug:
                    print(f"Segment is flat in dimension {flat_dim}, adding artificial thickness")
                
                # Add small noise in the flat dimension
                noise = np.zeros_like(points)
                noise[:, flat_dim] = np.random.normal(0, min_range/4, len(points))
                pcd.points = o3d.utility.Vector3dVector(points + noise)
            
            # Now compute OBB
            obb = pcd.get_oriented_bounding_box()

            self.extent = obb.extent
            a1, a2, a3 = self.extent[0] / 2.0, self.extent[1] / 2.0, self.extent[2] / 2.0

            self.pose = obb.R

            self.center = obb.center

            # Decide target number of points for final model (downsampling if needed)
            n_points = max(100, int(len(pcd.points) * self.downsample // 100))

            # Superquadric parameter generation x, y, z
            eta = (np.random.rand(n_points) - 0.5) * np.pi 
            eta *= 1.0
            eta = eta / 1.0 
            u = np.random.rand(n_points) * 2.0 - 1.0 
            eta = np.arcsin(np.clip(u, -1.0, 1.0))  
            omega = (np.random.rand(n_points) * 2.0 - 1.0) * np.pi 

            def sgn(x):
                return np.sign(x + 1e-15)

            ce, se = np.cos(eta), np.sin(eta)
            co, so = np.cos(omega), np.sin(omega)

            x = a1 * sgn(ce) * np.abs(ce)**e1 * sgn(co) * np.abs(co)**e2
            y = a2 * sgn(ce) * np.abs(ce)**e1 * sgn(so) * np.abs(so)**e2
            z = a3 * sgn(se) * np.abs(se)**e1

            # World transform
            V_local = np.stack([x, y, z], axis=1) 
            V_world = (self.pose @ V_local.T).T + self.center

            # Build point cloud directly
            sq_pcd = o3d.geometry.PointCloud()
            sq_pcd.points = o3d.utility.Vector3dVector(V_world.astype(np.float64))

            # Estimate normals once
            sq_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )

            return sq_pcd
        
        except Exception as e:
            print(f"superquadric [createSuperquadric] Error: {e}")
            return 1.0, 1.0

    def getSuperquadricMesh(self):
        return self.superquadric
    
    def getCenter(self):
        return self.center
    
    def getBBOXExtent(self):
        return self.extent
    
    def getSQPose(self):
        return self.pose
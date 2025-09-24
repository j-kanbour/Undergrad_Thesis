"""
    Grasp Generator

    author: Jayden Kanbour
    UNSW student_id: z5316799

    Description:    given a point cloud model of an object, this class will generate 
                    possible grasps points, using a series of criteria to determine
                    optimal grasp points

    Input:
        - Superquadric object
        - orientation: {top, top2, front, front-vertical}

    Output:
        - pose : center point of contact on the object surface, pointign away from the object
        - grasp point 1
        - grasp point 2 

"""

import os
import sys
import rospy
import numpy as np 
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_matrix

module_path = os.environ.get("UNSW_WS")
sys.path.append(module_path + "/PLANNING/action_server/src/grasp_code")

import grasp_checks
class Grasps:
    def __init__(self, sq, orientation=None, gripper_w = 0.5, gripper_d = 0.5):
        #blinky: depth=0.0666, width=0.236

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)

        #extract necessary information from superquadric object
        #sq = sq.getSuperquadricAsPCD()
        
        # self.depth_masked = sq.getRawData()["masked_depth"]
        
        # #extracts camera info
        # self.camera_info = sq.getRawData()["camera_info"]
        # self.K = np.array(self.camera_info.K).reshape(3, 3)
        # self.fx = self.K[0, 0]
        # self.fy = self.K[1, 1]
        # self.cx = self.K[0, 2]
        # self.cy = self.K[1, 2]
        # self.w = self.camera_info.width
        # self.h = self.camera_info.height
        
        # self.depth_scale = 0.001 #??

        # self.object_pcd = sq.getPCD().getPCD()

        #generate and select best grasp
        self.orientation = orientation
        self.allGrasps = self.generateGrasps(sq, orientation, gripper_d, gripper_w) #{grasp: score}
        self.selectedGrasps = self.selectGrasps(sq, orientation)

    #generate num_grasps possible grasps
    def generateGrasps(self, sq, num_grasps=50, dist=1.0):

        #use the grasp orientation to to limit the superquadrics being searched for grasps

        try:
            points = np.asarray(sq.points)
            normals = np.asarray(sq.normals)
            length_points = len(points)
            candidate_grasps = {}  # {geometry.pose: score}

            for i in range(1, len(points), 50):
                point1 = points[i]
                normal1 = normals[i]
                second_index = length_points - i
                point2 = points[second_index]
                normal2 = normals[second_index]

                # Full grasp info
                grasp_pose = {
                    "score": 0,
                    "index_i": i,
                    "index_j": second_index,
                    "point_i": point1.copy(),
                    "point_j": point2.copy(),
                    "point_i_normals": normal1.copy(),
                    "point_j_normals": normal2.copy(),
                }

                #discotinue checks if 1 fails
                # Run checks and score
                gripper_score = grasp_checks.checkGripper(grasp_pose, sq)
                if gripper_score < 100: break
                
                antipodal_score = grasp_checks.checkAntipodal(grasp_pose, normal_threshold=10) 
                if antipodal_score < 70: break
                #collision_score = grasp_checks.checkCollision(grasp_pose, self.depth_masked, self.camera_info, collision_threshold=0.05)
                # total_score = grip_score + antipodal_score #+ collision_score
                # if grip_score > 50 and antipodal_score > 50: # and collision_score > 50:

                #TODO: bind better method than scoreing: affordances
                #this will overwrite ang grasps of the same score
                total_score = gripper_score + antipodal_score
                grasp_pose["score"] = total_score

                candidate_grasps.insert() = grasp_pose

                if len(candidate_grasps) >= num_grasps: break

            print(f'number of candidate grasps: {len(candidate_grasps)}')
            return candidate_grasps

        except Exception as e:
            print(f"[generateGrasps] Error: {e}")
            return None

    def generatPose(self, grasp_pose):
        # Extract and cast to float64 to prevent dtype errors
        point1 = np.array(grasp_pose["point_i"], dtype=np.float64)
        point2 = np.array(grasp_pose["point_j"], dtype=np.float64)
        normal1 = np.array(grasp_pose["point_i_normals"], dtype=np.float64)
        normal2 = np.array(grasp_pose["point_j_normals"], dtype=np.float64)

        # Define grasp line and midpoint
        vec = point2 - point1
        grasp_line = vec / np.linalg.norm(vec)
        midpoint = (point1 + point2) / 2.0

        # Rescale endpoints to fixed 20cm grasp (±0.1m)
        half_length = 0.1
        point1 = midpoint - half_length * grasp_line
        point2 = midpoint + half_length * grasp_line

        # Choose pose vector (approach direction)
        # need to re-consider grasp instructions top/front/blank(most optimal)

        if self.orientation in ['top', 'top2']:
            pose_vector = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        elif self.orientation in ['front', 'front-vertical']:
            ground_normal = np.array([0.0, 0.0, 1.0])
            pose_vector = np.cross(grasp_line, ground_normal)
            if np.linalg.norm(pose_vector) < 0.1:
                fallback_axis = np.array([1.0, 0.0, 0.0]) if abs(grasp_line[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                pose_vector = np.cross(grasp_line, fallback_axis)
            pose_vector[2] = 0.0
            pose_vector = pose_vector / np.linalg.norm(pose_vector)
        else:
            avg_normal = (normal1 + normal2) / 2.0
            pose_vector = avg_normal / np.linalg.norm(avg_normal)

        # Enforce perpendicularity (except top views)
        if self.orientation not in ['top', 'top2']:
            dot_product = np.dot(pose_vector, grasp_line)
            if abs(dot_product) > 0.1:
                pose_vector = pose_vector - dot_product * grasp_line
                pose_vector = pose_vector / np.linalg.norm(pose_vector)

        # Build orthonormal rotation matrix (z = approach, y = grasp)
        z_axis = pose_vector
        y_axis = grasp_line
        x_axis = np.cross(y_axis, z_axis)
        y_axis = np.cross(z_axis, x_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        rot = np.eye(4)  # ← Change to 4x4 for quaternion_from_matrix
        rot[:3, 0] = x_axis
        rot[:3, 1] = y_axis
        rot[:3, 2] = z_axis

        quat = quaternion_from_matrix(rot)

        # Final pose
        x = float(midpoint[0])
        y = float(midpoint[1])
        z = np.clip(float(midpoint[2]), 0.02, 2.00)

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = self.camera_info.header.frame_id
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]

        return pose_stamped


    def selectGrasps(self):
        if not self.allGrasps:
            rospy.logwarn("[selectGrasps] No grasps were generated.")
            return None

        try:
            # Sort all grasp candidates by descending score
            sorted_grasps = sorted(self.allGrasps.items(), key=lambda item: item[0], reverse=True)

            for score, grasp_pose in sorted_grasps:
                try:
                    pose_stamped = self.generatPose(grasp_pose)
                    if pose_stamped is not None:
                        rospy.loginfo(f"[selectGrasps] Selected grasp with score: {score:.2f}")
                        return pose_stamped
                except Exception as e:
                    rospy.logwarn(f"[selectGrasps] Skipped invalid grasp (score {score:.2f}): {e}")
                    continue  # Try next best grasp

            rospy.logwarn("[selectGrasps] No valid grasp poses after checking all candidates.")
            return None

        except Exception as e:
            rospy.logerr(f"[selectGrasps] Fatal error during selection: {e}")
            return None

    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps
    
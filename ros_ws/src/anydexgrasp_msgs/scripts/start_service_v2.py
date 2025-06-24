#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Provides a ROS service to convert point clouds to Inspire hand grasp poses using AnyDexGrasp V3.
"""
from typing import Union
import rospy
import numpy as np
import sys
import torch
import copy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_srvs.srv import Trigger, TriggerResponse
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# --- V3 Imports ---
from inference_v3.config import get_config
from inference_v3.utils.net_utils import get_net
from inference_v3.utils.data_utils import get_inspire_model, load_meshes_pointcloud
from inference_v3.utils.grasp_utils import flip_ggarray, get_graspgroup_features, get_inspire_depth_type, select_grasp_type
from inference_v3.utils.visualization_utils import visualize_grasp_proposals
from inference_v3.main import get_ggarray_features # Using the main function for convenience

from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup, InspireHandRGrasp
from graspnetAPI import GraspGroup, Grasp
from adg_utils.collision_detector import ModelFreeCollisionDetectorMultifinger

# Custom service messages
try:
    from anydexgrasp_msgs.srv import GraspPlanning, GraspPlanningResponse
    from anydexgrasp_msgs.msg import GraspPose
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("Custom message types 'anydexgrasp_msgs' not found. Using standard ROS messages.")
    CUSTOM_MSGS_AVAILABLE = False
    # Define placeholder for type hinting if msgs are not available
    class GraspPose: pass


class InspireGraspPlanningService:
    """ROS service for Inspire hand grasp planning from point clouds (V3 Version)"""

    def __init__(self):
        rospy.init_node("inspire_grasp_planning_service", anonymous=False)

        # 1. Load V3 configurations
        self.cfgs = get_config()

        # 2. Get ROS parameters and override V3 configs where necessary
        # This allows for flexibility via launch files
        self.cfgs.checkpoint_path = rospy.get_param(
            "~checkpoint_path", self.cfgs.checkpoint_path
        )
        self.score_threshold = rospy.get_param(
            "~score_threshold", 0.85
        )
        self.max_grasps = rospy.get_param("~max_grasps", 10)
        
        # Workspace filter parameters (you can also add these to your V3 config file)
        self.workspace_mask = [
            rospy.get_param("~workspace/x_min", 0.57), rospy.get_param("~workspace/x_max", 1.0),
            rospy.get_param("~workspace/y_min", -0.2), rospy.get_param("~workspace/y_max", 0.2),
            rospy.get_param("~workspace/z_min", -0.6-1),  rospy.get_param("~workspace/z_max", -0.6)
        ]

        rospy.loginfo("Initializing Inspire Grasp Planning Service with V3 backend...")
        rospy.loginfo(f"Using checkpoint: {self.cfgs.checkpoint_path}")

        # 3. Initialize models using V3 functions
        self._initialize_models()

        # 4. Create ROS service
        self.grasp_service = rospy.Service(
            "~plan_grasps", self._get_service_type(), self._handle_grasp_planning
        )

        # 5. Create publisher for visualization
        self.grasp_poses_pub = rospy.Publisher(
            "~grasp_poses", PoseStamped, queue_size=10
        )

        rospy.loginfo("Inspire Grasp Planning Service initialized successfully!")
        rospy.loginfo(f"Service available at: {self.grasp_service.resolved_name}")

    def _get_service_type(self):
        """Gets the service type, falling back to Trigger if custom messages are unavailable"""
        if CUSTOM_MSGS_AVAILABLE:
            return GraspPlanning
        else:
            rospy.logwarn("Using Trigger service as fallback.")
            return Trigger

    def _initialize_models(self):
        """Initializes the neural network and gripper models using V3 functions."""
        try:
            rospy.loginfo("Loading GraspNet model (V3)...")
            self.net = get_net(self.cfgs.checkpoint_path, use_v2=self.cfgs.use_graspnet_v2, half_views=self.cfgs.half_views)
            rospy.loginfo("GraspNet model loaded.")

            rospy.loginfo("Loading Inspire gripper models and meshes (V3)...")
            self.inspire_models = get_inspire_model(self.cfgs.inspire_model_path)
            self.meshes_pcls = load_meshes_pointcloud(self.cfgs.inspire_mesh_json_path, self.cfgs.INSPIREHANDR_VOXElGRID)
            rospy.loginfo("Gripper-specific models loaded.")

            # Collision detector will be created per request with the downsampled point cloud
            self.collision_detector = None
        except Exception as e:
            rospy.logerr(f"Failed to initialize models: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _handle_grasp_planning(self, req):
        """Handles the grasp planning request."""
        try:
            rospy.loginfo("Received grasp planning request.")
            
            # Extract point cloud from the request
            if CUSTOM_MSGS_AVAILABLE and hasattr(req, "pointcloud"):
                points = self._extract_points_from_pointcloud2(req.pointcloud)
                self.frame_id = req.pointcloud.header.frame_id
            else:
                rospy.loginfo("Waiting for point cloud on /camera/depth/color/points...")
                pointcloud_msg = rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=10.0)
                points = self._extract_points_from_pointcloud2(pointcloud_msg)
                self.frame_id = pointcloud_msg.header.frame_id

            rospy.loginfo(f"Received and filtered point cloud with {len(points)} points.")

            # Execute grasp planning
            grasp_poses = self._plan_grasps(points)
            for grasp_pose in grasp_poses:
                grasp_pose.header.frame_id = self.frame_id
            # Create and return the response
            response = self._create_response(req, grasp_poses)
            
            # Publish poses for visualization in RViz
            for grasp_pose in grasp_poses:
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.header.frame_id = self.frame_id
                pose_stamped.pose = grasp_pose.pose
                self.grasp_poses_pub.publish(pose_stamped)
                # rospy.sleep(0.1)

            rospy.loginfo(f"Successfully generated and published {len(grasp_poses)} grasp poses.")
            return response

        except Exception as e:
            rospy.logerr(f"Error in grasp planning: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(req, str(e))

    def _create_response(self, req, grasp_poses):
        """Creates a success response message."""
        if CUSTOM_MSGS_AVAILABLE and hasattr(req, "pointcloud"):
            response = GraspPlanningResponse()
            response.success = len(grasp_poses) > 0
            response.message = f"Generated {len(grasp_poses)} grasp poses."
            response.grasp_poses = grasp_poses
        else:
            response = TriggerResponse()
            response.success = len(grasp_poses) > 0
            response.message = f"Generated {len(grasp_poses)} grasp poses."
        return response

    def _create_error_response(self, req, error_message):
        """Creates an error response message."""
        if CUSTOM_MSGS_AVAILABLE and hasattr(req, "pointcloud"):
            response = GraspPlanningResponse()
            response.success = False
            response.message = f"Error: {error_message}"
            response.grasp_poses = []
        else:
            response = TriggerResponse()
            response.success = False
            response.message = f"Error: {error_message}"
        return response

    def _extract_points_from_pointcloud2(self, pointcloud_msg):
        """Extracts point cloud data from a PointCloud2 message and applies a workspace filter."""
        points_list = list(pc2.read_points(
            pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True
        ))
        
        if not points_list:
            raise ValueError("No valid points found in point cloud message.")
            
        points = np.array(points_list, dtype=np.float32)
        np.savetxt("/data/shiqi/AnyDexGrasp/origin.txt", points)

        # Apply workspace filtering
        x_min, x_max, y_min, y_max, z_min, z_max = self.workspace_mask
        mask = (
            (points[:, 0] > x_min) & (points[:, 0] < x_max) &
            (points[:, 1] > y_min) & (points[:, 1] < y_max) &
            (points[:, 2] > z_min) & (points[:, 2] < z_max)
        )
        filtered_points = points[mask]

        if len(filtered_points) == 0:
            rospy.logwarn(f"Point cloud stats before filtering: "
                          f"x:[{np.min(points[:,0]):.2f}, {np.max(points[:,0]):.2f}], "
                          f"y:[{np.min(points[:,1]):.2f}, {np.max(points[:,1]):.2f}], "
                          f"z:[{np.min(points[:,2]):.2f}, {np.max(points[:,2]):.2f}]")
            raise ValueError("No points remain after workspace filtering.")
            
        return filtered_points

    def _plan_grasps(self, points: np.ndarray):
        """Executes the core grasp planning pipeline using V3 functions."""
        
        # 1. Generate grasp proposals with data augmentation
        rospy.loginfo("Step 1: Generating grasp proposals with augmentation...")
        ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(self.net, points, None)
        if ggarray is None:
            rospy.logwarn("No grasp proposals generated from the network.")
            return []
        rospy.loginfo(f"Generated {len(ggarray)} raw grasp proposals.")

        # 2. Process proposals: convert to numpy, flip, filter, and sort
        rospy.loginfo("Step 2: Processing and filtering raw proposals...")
        ggarray = ggarray.cpu().numpy()
        grasp_features = grasp_features.cpu().numpy()

        ggarray, if_flip = flip_ggarray(ggarray)
        grasp_features = np.c_[grasp_features, if_flip]
        
        # For Inspire hand (asymmetric), remove the flipped grasps
        ggarray = ggarray[~np.array(if_flip)]
        grasp_features = grasp_features[~np.array(if_flip)]
        
        # Sort by score and keep the top proposals for the next stage
        source_index = ggarray[:, 0].argsort()[::-1][:2500] # Corresponds to V3 script
        ggarray = ggarray[source_index]
        grasp_features = grasp_features[source_index]
        rospy.loginfo(f"Kept {len(ggarray)} proposals after flipping and sorting.")
        
        import pdb
        pdb.set_trace()

        # 3. Predict multi-finger grasp types and depths
        rospy.loginfo("Step 3: Predicting multi-finger grasp type and depth...")
        grasp_features_dic = get_graspgroup_features(grasp_features, sinput)
        inspire_depth, inspire_type, scores, ggarray, grasp_features = \
            get_inspire_depth_type(self.inspire_models, grasp_features_dic, ggarray, grasp_features=grasp_features,
                                   num_inspire_depth=self.cfgs.NUM_OF_INSPIRE_DEPTH,
                                   num_inspire_type=self.cfgs.NUM_OF_INSPIRE_TYPE)
        
        # Filter by score threshold
        mask = (scores > self.score_threshold)
        ggarray, grasp_features, inspire_depth, inspire_type, scores = (
            ggarray[mask], grasp_features[mask], inspire_depth[mask], inspire_type[mask], scores[mask]
        )
        if len(ggarray) == 0:
            rospy.logwarn(f"No grasps passed the score threshold of {self.score_threshold}.")
            return []
        rospy.loginfo(f"{len(ggarray)} grasps remain after score filtering.")

        # 4. Create GraspGroup objects and refine poses
        rospy.loginfo("Step 4: Creating full gripper grasp objects...")
        two_fingers_gg = GraspGroup(ggarray)
        gripper_gg = InspireHandRGraspGroup()
        gripper_gg.set_grasp_min_width(self.cfgs.MIN_GRASP_WIDTH)
        gripper_gg.from_graspgroup(two_fingers_gg, inspire_type, self.cfgs.inspire_mesh_json_path)
        gripper_gg.scores = scores
        gripper_gg.depths += inspire_depth + self.cfgs.INSPIREHANDR_DEFAULT_DEPTH
        
        # Select best grasp type for each location
        index_type = select_grasp_type(gripper_gg, self.cfgs.NUM_OF_INSPIRE_TYPE)
        gripper_gg = gripper_gg[index_type]
        two_fingers_gg = two_fingers_gg[index_type]
        
        if len(gripper_gg) == 0:
            rospy.logwarn("No grasps left after type selection.")
            return []
        rospy.loginfo(f"{len(gripper_gg)} grasps remain after type selection.")
        
        # Sort again by final scores
        index_score = np.argsort(gripper_gg.scores)[::-1]
        gripper_gg = gripper_gg[index_score]
        two_fingers_gg = two_fingers_gg[index_score]

        # 5. Collision Detection
        rospy.loginfo("Step 5: Performing collision detection...")
        mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
        coll_gg, coll_tf_gg, empty_mask, _ = mfcdetector.detect(
            gripper_gg, two_fingers_gg, self.cfgs.inspire_mesh_json_path, self.meshes_pcls,
            min_grasp_width=self.cfgs.MIN_GRASP_WIDTH,
            VoxelGrid=self.cfgs.INSPIREHANDR_VOXElGRID,
            approach_dist=0.06, # from V3 script
            collision_thresh=0,
            adjust_gripper_centers=True
        )
        
        gripper_gg_final = coll_gg[empty_mask]
        two_fingers_gg_final = coll_tf_gg[empty_mask]

        if len(gripper_gg_final) == 0:
            rospy.logwarn("No grasps remaining after collision detection.")
            return []
        rospy.loginfo(f"{len(gripper_gg_final)} grasps remain after collision detection.")
        
        # workspace_filter
        translations = gripper_gg_final.translations
        x_min, x_max, y_min, y_max, z_min, z_max = self.workspace_mask
        x_cond = (translations[:, 0] >= (x_min)) & (translations[:, 0] <= (x_max))
        y_cond = (translations[:, 1] >= (y_min)) & (translations[:, 1] <= (y_max))
        # z_cond = (translations[:, 2] >= z_min)# & (translations[:, 2] <= z_max)
        # 组合所有条件
        valid_mask = x_cond & y_cond
        # 获取有效索引
        ws_mask = np.where(valid_mask)[0].tolist()
        # gripper_gg_final = gripper_gg_final[ws_mask]
        # two_fingers_gg_final = two_fingers_gg_final[ws_mask]
        
        if len(gripper_gg_final) == 0:
            rospy.logwarn("No grasps remaining after ws detection.")
            return []
        rospy.loginfo(f"{len(gripper_gg_final)} grasps remain after ws detection.")
        
        # 6. Final Selection and Conversion
        rospy.loginfo("Step 6: Selecting top grasps and converting to ROS format.")
        index_score_post = np.argsort(gripper_gg_final.scores)[::-1][:self.max_grasps]#[0:1]
        top_grasps = gripper_gg_final[index_score_post]
        top_tf_grasps = two_fingers_gg_final[index_score_post]

        # Optional: Visualize final grasp proposals in Open3D
        if self.cfgs.visualize_final_grasps: # Add this to your config if you want
            vis_cloud = o3d.geometry.PointCloud()
            vis_cloud.points = o3d.utility.Vector3dVector(points)
            visualize_grasp_proposals(vis_cloud, top_tf_grasps, top_grasps, self.cfgs, "Final Top Grasp Proposals", "/data/shiqi/AnyDexGrasp/grasp_result.ply")

        grasp_poses = [self._convert_grasp_to_pose(g) for g in top_grasps]
        
        return grasp_poses

    def _convert_grasp_to_pose(self, grasp: Union[InspireHandRGrasp, Grasp]) -> GraspPose:
        """Converts an internal grasp object to a ROS GraspPose message."""
        
        pose_msg = Pose()
        # Position
        pose_msg.position = Point(
            x=float(grasp.translation[0]),
            y=float(grasp.translation[1]),
            z=float(grasp.translation[2]),
        )
        # Orientation
        rotation_matrix = grasp.rotation_matrix.reshape(3, 3)
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]
        pose_msg.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
        )

        if not CUSTOM_MSGS_AVAILABLE:
            return pose_msg

        # If custom messages are available, populate the full message
        grasp_pose_msg = GraspPose()
        grasp_pose_msg.pose = pose_msg
        grasp_pose_msg.score = grasp.score
        # Ensure angles are in the correct format
        grasp_pose_msg.angles = np.array(grasp.angle, dtype=np.int32)
        
        return grasp_pose_msg

    def run(self):
        """Runs the ROS service node."""
        rospy.loginfo("Inspire Grasp Planning Service is running...")
        rospy.loginfo("Waiting for grasp planning requests...")
        rospy.spin()

if __name__ == "__main__":
    try:
        service = InspireGraspPlanningService()
        service.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Inspire Grasp Planning Service interrupted.")
    except Exception as e:
        rospy.logerr(f"Failed to start service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
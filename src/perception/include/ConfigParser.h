#ifndef CONFIGPARSER_HH_
#define CONFIGPARSER_HH_

#include "Utils.h"
#include "yaml-cpp/yaml.h"

class ConfigParser
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ConfigParser(){};
  ConfigParser(std::string cfg_file);
  ~ConfigParser();
  void parseYMLFile(std::string filepath);

  YAML::Node yml;

  Eigen::Matrix3f cam_intrinsic;
  Eigen::Matrix4f endeffector2global, palm_in_baselink, handbase_in_palm, refine_calibration_offset, cam1_in_leftarm, leftarm_in_base;
  std::string msg_rgb_topic, msg_depth_topic;
  ros::NodeHandle nh;
  std::string rgb_topic, depth_topic, object_model_path;
  float leaf_size, distance_to_table, radius, super4pcs_overlap, super4pcs_delta, super4pcs_max_normal_difference, super4pcs_max_color_distance, particle_filter_delta, particle_filter_epsilon, particle_filter_resample_likelihood_thres, particle_filter_max_dist,pose_estimator_high_confidence_thres, gripper_min_dist;
  int max_iterations, min_number, LCCP_k_factor,supervoxel_refine_times, super4pcs_sample_size, particle_filter_particle_num, particle_filter_particle_max_num, super4pcs_max_time_seconds;
  std::string rgb_path, depth_path, object_mesh_path;
  float pose_estimator_wrong_ratio, normal_radius;


  std::string cam_in_world_file;
  Eigen::Matrix4f cam_in_world;

};


#endif
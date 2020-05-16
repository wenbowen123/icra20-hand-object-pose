#ifndef HAND_HH_
#define HAND_HH_


#include "Utils.h"
#include "ConfigParser.h"


class FingerProperty
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  float _min_x, _min_y, _min_z, _max_x, _max_y, _max_z;    // Extreme points in init status
  float _stride_z;
  Eigen::MatrixXf _hist_alongz;   // 6XN Each bin, <min_pt, max_pt>
  int _num_division;

  FingerProperty();
  FingerProperty(PointCloudRGBNormal::Ptr model, int num_division);
  ~FingerProperty();
  int getBinAlongZ(float z);
};




#include "optim.hpp"

class Hand
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Hand();
  Hand(ConfigParser *cfg1, const Eigen::Matrix3f &cam_K);
  ~Hand();
  void setCurScene(const cv::Mat &depth_meters, PointCloudRGBNormal::Ptr scene_organized, PointCloudRGBNormal::Ptr scene_hand_region, const Eigen::Matrix4f &handbase_in_cam);
  void reset();
  void parseURDF();
  void addComponent(std::string name, std::string parent_name, PointCloudRGBNormal::Ptr cloud, pcl::PolygonMesh::Ptr mesh, pcl::PolygonMesh::Ptr convex_mesh, const Eigen::Matrix4f &tf_in_parent, const Eigen::Matrix4f &tf_self);
  void getTFHandBase(std::string cur_name, Eigen::Matrix4f &tf_in_handbase);
  void makeHandCloud();
  void makeHandMesh();
  void printComponents();

  template<class CloudT>
  void removeSurroundingPoints(boost::shared_ptr<CloudT> scene, boost::shared_ptr<CloudT> scene_out, float dist_thres);
  void initPSO();
  bool matchOneComponentPSO(std::string model_name, float min_angle, float max_angle, bool use_normal, float dist_thres, float normal_angle_thres, float least_match);
  void handbaseICP(PointCloudRGBNormal::Ptr scene_organized);

  std::map<std::string, PointCloudRGBNormal::Ptr> _clouds;
  std::map<std::string, pcl::PolygonMesh::Ptr> _meshes, _convex_meshes;  //NOTE: _convex_meshes for collision
  std::map<std::string, std::string> _parent_names;   // base parent is world
  std::map<std::string, Eigen::Matrix4f, std::less<std::string>, Eigen::aligned_allocator<std::pair<std::string, Eigen::Matrix4f> > > _tf_in_parent;
  std::map<std::string, boost::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> > > _kdtrees;    // Component kdtree in handbase frame
  std::map<std::string, std::vector<int> > _matched_scene_indices;   // indices on _scene_sampled that is near to the components under some thres
  std::map<std::string, FingerProperty> _finger_properties;
  // Finger pose before transform to parent. This is to be estimated
  std::map<std::string, Eigen::Matrix4f, std::less<std::string>, Eigen::aligned_allocator<std::pair<std::string, Eigen::Matrix4f> > > _tf_self;
  std::map<std::string, bool> _component_status;
  std::map<std::string, float> _finger_angles;

  Eigen::Matrix4f _handbase_in_cam;
  PointCloudRGBNormal::Ptr _hand_cloud;    // whole hand cloud in handbase
  cv::Mat _depth_meters;
  Eigen::Matrix3f _cam_K;
  PointCloudRGBNormal::Ptr _scene_organized, _scene_sampled, _scene_hand_region;

  ConfigParser *cfg;
  optim::ArgPasser _pso_args;
  optim::algo_settings_t _pso_settings;

};


class HandT42:public Hand
{
public:
  HandT42();
  HandT42(ConfigParser *cfg1, const Eigen::Matrix3f &cam_K);
  ~HandT42();

  template<class PointT, bool use_normal>
  void removeSurroundingPointsAndAssignProbability(boost::shared_ptr<pcl::PointCloud<PointT> > scene, boost::shared_ptr<pcl::PointCloud<PointT> > scene_out, float dist_thres);


  void fingerICP();
  void adjustHandHeight();



};

#endif
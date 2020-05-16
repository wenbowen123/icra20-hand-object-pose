#ifndef POSE_ESTIMATOR_HH_PERCEPTION
#define POSE_ESTIMATOR_HH_PERCEPTION

#include "Utils.h"
#include "ConfigParser.h"
#include "Renderer.h"
#include "Hand.h"
#include "SDFchecker.h"
#include "PoseHypo.h"

template<class PointT>
class PoseEstimator
{
public:
  PoseEstimator(ConfigParser *cfg1,  boost::shared_ptr<pcl::PointCloud<PointT> > model, boost::shared_ptr<pcl::PointCloud<PointT> > model001, const Eigen::Matrix3f &cam_K);
  ~PoseEstimator();
  void reset();
  void setCurScene(boost::shared_ptr<pcl::PointCloud<PointT> > scene, PointCloudRGBNormal::Ptr cloud_withouthand_raw, boost::shared_ptr<pcl::PointCloud<PointT> > object_segment, const cv::Mat &rgb, const cv::Mat &depth_meters);
  bool runSuper4pcs(const std::map<std::vector<int>, std::vector<std::pair<int, int> > > &ppfs);
  void refineByICP();
  void selectBest(PoseHypo &best_hypo);
  void clusterPoses(float angle_diff, float dist_diff, bool assign_id);
  void projectCloudAndCompare(boost::shared_ptr<pcl::PointCloud<PointT> > cloud, cv::Mat &projection, float &wrong_ratio, int id);
  // void register(std::string dir, std::string name);
  void rejectByCollisionOrNonTouching(HandT42* hand);
  void registerMesh(std::string mesh_dir, std::string name, const Eigen::Matrix4f &pose);
  void registerHandMesh(Hand *hand);
  void rejectByRender(float projection_thres, HandT42 *hand);




protected:
  boost::shared_ptr<pcl::PointCloud<PointT> > _model, _model001, _scene, _scene_high_confidence, _object_segment;
  PointCloudRGBNormal::Ptr _cloud_withouthand_raw;
  Eigen::Matrix3f _cam_K;
  cv::Mat _depth_meters, _rgb;
  ConfigParser *cfg;
  std::vector<PoseHypo,Eigen::aligned_allocator<PoseHypo> > _pose_hypos;
  Renderer _renderer;
  pcl::PointXYZRGBNormal _model_center_init;
  pcl::PolygonMesh::Ptr _obj_mesh;
  float _model_radius, _smallest_dim, _ob_diameter;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  SDFchecker _sdf;

};



#endif
#ifndef COMMON_IO__HH
#define COMMON_IO__HH

// STL
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <time.h>
#include <queue>
#include <climits>
#include <boost/assign.hpp>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <math.h>
#include <boost/format.hpp>
#include <numeric>
#include <thread>
#include <omp.h>
// Basic ROS
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

// Basic PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/pca.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>

// For IO
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>

// For OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PointStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/ppf.h>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <pcl/features/integral_image_normal.h>
#include <pcl/tracking/normal_coherence.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/features/principal_curvatures.h>
#include <boost/serialization/array.hpp>
#define EIGEN_DENSEBASE_PLUGIN "EigenDenseBaseAddons.h"





#define SR300_DEPTH_UNIT 0.001   // one unit equals how many meters


// definitions
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudRGBNormal;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudNormal;
typedef pcl::PointCloud<pcl::PointSurfel> PointCloudSurfel;
typedef pcl::PointCloud<pcl::PrincipalCurvatures> PointCloudCurvatures;


// #define DBG_ICP
// #define DBG_PHYSICS

// Declaration for common utility functions
namespace Utils
{
void delimitString(std::string str, char dilimiter, std::vector<float> &v);
float rotationGeodesicDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2);
Eigen::Quaternionf averageQuaternions(const std::vector<Eigen::Quaternionf> &qs) ;
template<class PointT>
void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<PointT>> objCloud);
void readDepthImage(cv::Mat &depthImg, std::string path);
void writeDepthImage(cv::Mat &depthImg, std::string path);
void transformPolygonMesh(pcl::PolygonMesh::Ptr mesh_in, pcl::PolygonMesh::Ptr mesh_out, Eigen::Matrix4f transform);
template<class PointT>
float runICP(boost::shared_ptr<pcl::PointCloud<PointT> > pclSegment,
                     boost::shared_ptr<pcl::PointCloud<PointT> > pclModel,
                     Eigen::Matrix4f &offsetTransform, int max_iter=30, float rejection_angle=60, float max_corres_dist=0.01, float score_thres=1e-4);

void runICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel,
                Eigen::Matrix4f &offsetTransform, float max_corres_dist);
void readDirectory(const std::string& name, std::vector<std::string>& v);
template<class T>
Eigen::Matrix<T,3,3> eulerToRotationMatrix(T roll, T pitch, T yaw);
template<class PointType>
void downsamplePointCloud(boost::shared_ptr<pcl::PointCloud<PointType> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointType> > cloud_out, float vox_size);
template<class PointType>   // PointType must contain Normal
void calCloudNormal(boost::shared_ptr<pcl::PointCloud<PointType> > cloud, float radius);
template<class CloudT>
void calNormalRGBD(const Eigen::Matrix3f &cam_intrinsic, cv::Mat depth_image, boost::shared_ptr<CloudT> cloud);
template <class PointT>
void calNormalIntegralImage(boost::shared_ptr<pcl::PointCloud<PointT>> cloud, int method, float max_depth_change_factor, float smooth_size,bool depth_dependent_smooth);
template<class PointType>
bool runICP(boost::shared_ptr<pcl::PointCloud<PointType>> cloud1, boost::shared_ptr<pcl::PointCloud<PointType>> cloud2,
            Eigen::Matrix4f &trans);
template<class PointType>
void calNormalMLS(boost::shared_ptr<pcl::PointCloud<PointType>> cloud, float normal_radius);
template <typename T>
std::vector<int> vectorArgsort(const std::vector<T> &v, bool min_to_max);
template<class PointT>
float computeLCP(boost::shared_ptr<pcl::PointCloud<PointT> > scene, const std::vector<float> &scene_weights, boost::shared_ptr<pcl::PointCloud<PointT> > model, float dist_thres=5e-3, float angle_thres=15, bool use_normal=true, bool use_dot_score=false, bool use_reciprocal=false);
template<class PointT>
float computeAverageDistance(boost::shared_ptr<pcl::PointCloud<PointT> > scene, boost::shared_ptr<pcl::PointCloud<PointT> > model);
template <class PointT>
void removeAllNaNFromPointCloud(boost::shared_ptr<pcl::PointCloud<PointT> > cloud);
void parsePoseTxt(std::string filename, Eigen::Matrix4f &out);

} // namespace Utils

ostream& operator<< (ostream& os, const Eigen::Quaternionf q);



#endif
#include "Utils.h"
#include <vtkPolyLine.h>   //VTK include needed for drawing graph lines
#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkImageData.h>
#include <vtkImageFlip.h>




namespace enc = sensor_msgs::image_encodings;

namespace Utils
{

void delimitString(std::string str, char dilimiter, std::vector<float> &v)
{
  std::stringstream ss(str);
  std::string temp;
  v.clear();
  while(getline(ss,temp, dilimiter)) // delimiter as space
  {
    v.push_back(std::stof(temp));
  }
}


// Difference angle in radian
float rotationGeodesicDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2)
{
  return std::acos(((R1 * R2).trace()-1) / 2.0);
}


// return depth in meter
void readDepthImage(cv::Mat &depthImg, std::string path)
{
  cv::Mat depthImgRaw = cv::imread(path, CV_16UC1);
  depthImg = cv::Mat::zeros(depthImgRaw.rows, depthImgRaw.cols, CV_32FC1);
  for (int u = 0; u < depthImgRaw.rows; u++)
    for (int v = 0; v < depthImgRaw.cols; v++)
    {
      unsigned short depthShort = depthImgRaw.at<unsigned short>(u, v);  // 16bits
      float depth = (float)depthShort * SR300_DEPTH_UNIT;  //NOTE: 10000 for previously saved depth images
      if (depth>2.0 || depth<0.1)
      {
        depthImg.at<float>(u, v) = 0.0;
      }
      else
      {
        depthImg.at<float>(u, v) = depth;
      }

    }
}

/********************************* function: writeDepthImage *******************************************
	*******************************************************************************************************/
// depthImg: must be CV_32FC1, unit in meters
void writeDepthImage(cv::Mat &depthImg, std::string path)
{
  cv::Mat depthImgRaw = cv::Mat::zeros(depthImg.rows, depthImg.cols, CV_16UC1);
  for (int u = 0; u < depthImg.rows; u++)
    for (int v = 0; v < depthImg.cols; v++)
    {
      float depth = depthImg.at<float>(u, v) / SR300_DEPTH_UNIT;
      // std::cout<<"writeDepthImage:"<<depth<<"\n";
      unsigned short depthShort = (unsigned short)depth;
      depthImgRaw.at<unsigned short>(u, v) = depthShort;
    }
  cv::imwrite(path, depthImgRaw);
}

/********************************* function: convert3dOrganizedRGB<pcl::PointXYZRGB> *************************************
colImage: 8UC3
objDepth: 16UC1
	*******************************************************************************************************/
template<class PointT>
void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<PointT>> objCloud)
{
  const int imgWidth = objDepth.cols;
  const int imgHeight = objDepth.rows;

  objCloud->height = (uint32_t)imgHeight;
  objCloud->width = (uint32_t)imgWidth;
  objCloud->is_dense = false;
  objCloud->points.resize(objCloud->width * objCloud->height);

  const float bad_point = 0;  // this can cause the point cloud visualization problem !!

  for (int u = 0; u < imgHeight; u++)
    for (int v = 0; v < imgWidth; v++)
    {
      float depth = objDepth.at<float>(u, v);
      cv::Vec3b colour = colImage.at<cv::Vec3b>(u, v); // 3*8 bits
      if (depth > 0.1 && depth < 2.0)
      {
        objCloud->at(v, u).x = (float)((v - camIntrinsic(0, 2)) * depth / camIntrinsic(0, 0));
        objCloud->at(v, u).y = (float)((u - camIntrinsic(1, 2)) * depth / camIntrinsic(1, 1));
        objCloud->at(v, u).z = depth;
        objCloud->at(v, u).b = colour[0];
        objCloud->at(v, u).g = colour[1];
        objCloud->at(v, u).r = colour[2];
      }
      else
      {
        objCloud->at(v, u).x = bad_point;
        objCloud->at(v, u).y = bad_point;
        objCloud->at(v, u).z = bad_point;
        objCloud->at(v, u).b = 0;
        objCloud->at(v, u).g = 0;
        objCloud->at(v, u).r = 0;
      }
    }
}
template void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> objCloud);
template void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal>> objCloud);


void transformPolygonMesh(pcl::PolygonMesh::Ptr mesh_in, pcl::PolygonMesh::Ptr mesh_out, Eigen::Matrix4f transform)
{
  PointCloudRGB::Ptr cloud_in(new PointCloudRGB);
  PointCloudRGB::Ptr cloud_out(new PointCloudRGB);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);
  pcl::transformPointCloud(*cloud_in, *cloud_out, transform);
  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return;
}

/********************************* function: runICP **********************************************
	*******************************************************************************************************/
//@segment: source
//@model: target
void runICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel,
                Eigen::Matrix4f &offsetTransform, float max_corres_dist)
{
  pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
  icp.setUseReciprocalCorrespondences(true);
  icp.setMaximumIterations(100);
  icp.setRANSACIterations(100);
  icp.setMaxCorrespondenceDistance(max_corres_dist);
  icp.setInputCloud(pclSegment);
  icp.setInputTarget(pclModel);
  // pcl::registration::CorrespondenceRejectorSurfaceNormal::Ptr rej (new pcl::registration::CorrespondenceRejectorSurfaceNormal);
  // rej->setThreshold (std::cos(90/180.0*M_PI));
  // icp.addCorrespondenceRejector (rej);
  pcl::PointCloud<pcl::PointXYZRGBNormal> Final;
  icp.align(Final);

  if (icp.hasConverged())
  {
    offsetTransform = icp.getFinalTransformation();
  }
  else
  {
    std::cout << "ICP did not converge. set to identity" << std::endl;
    offsetTransform << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
  }
}

//@cloud1: source
//@cloud2: target
template<class PointT>
bool runICP(boost::shared_ptr<pcl::PointCloud<PointT>> cloud1, boost::shared_ptr<pcl::PointCloud<PointT>> cloud2, Eigen::Matrix4f &trans)
{
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setUseReciprocalCorrespondences(true);
  icp.setMaximumIterations(100);
  icp.setRANSACIterations(100);
  icp.setInputSource(cloud1);
  icp.setInputTarget(cloud2);
  pcl::PointCloud<PointT> Final;
  icp.align(Final);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore(0.01) << std::endl;
  trans = icp.getFinalTransformation();

  return icp.hasConverged();
}

/********************************* function: runICP *******************************************
	*******************************************************************************************************/
template<class PointT>
float runICP(boost::shared_ptr<pcl::PointCloud<PointT> > pclSegment, boost::shared_ptr<pcl::PointCloud<PointT> > pclModel,         Eigen::Matrix4f &offsetTransform, int max_iter, float rejection_angle, float max_corres_dist, float score_thres)
{

  PointCloudNormal::Ptr modelCloud(new PointCloudNormal);
  PointCloudNormal::Ptr segmentCloud(new PointCloudNormal);
  PointCloudNormal::Ptr segCloudTrans(new PointCloudNormal);
  copyPointCloud(*pclModel, *modelCloud);
  copyPointCloud(*pclSegment, *segmentCloud);
  std::vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*modelCloud, *modelCloud, indices);
  pcl::removeNaNNormalsFromPointCloud(*segmentCloud, *segmentCloud, indices);
  pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> reg;
  pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>::Ptr trans_lls (
      new pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>);
  pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal>::Ptr cens (
    new pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal>);
  pcl::registration::CorrespondenceRejectorSurfaceNormal::Ptr rej1 (new pcl::registration::CorrespondenceRejectorSurfaceNormal);
  rej1->setThreshold (std::cos(rejection_angle/180.0*M_PI));   // we dont need to add src and tgt here!
  reg.convergence_criteria_->setRelativeMSE(1e-10);
  reg.convergence_criteria_->setAbsoluteMSE(1e-6);
  reg.setInputSource (segmentCloud);
  reg.setInputTarget (modelCloud);
  reg.setCorrespondenceEstimation (cens);
  reg.setTransformationEstimation (trans_lls);
  reg.addCorrespondenceRejector (rej1);
  reg.setMaximumIterations (max_iter);
  reg.setMaxCorrespondenceDistance (max_corres_dist);
  reg.align (*segCloudTrans);

  if (reg.hasConverged())
  {
    offsetTransform = reg.getFinalTransformation();
  }
  else
  {
    offsetTransform.setIdentity();
  }

  return reg.getFitnessScore(score_thres);

}
template float runICP<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > pclSegment, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > pclModel, Eigen::Matrix4f &offsetTransform, int max_iter, float rejection_angle, float max_corres_dist, float score_thres);
template float runICP<pcl::PointSurfel>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > pclSegment, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > pclModel, Eigen::Matrix4f &offsetTransform, int max_iter, float rejection_angle, float max_corres_dist, float score_thres);


void readDirectory(const std::string& name, std::vector<std::string>& v)
{
  v.clear();
  DIR *dirp = opendir(name.c_str());
  struct dirent *dp;
  while ((dp = readdir(dirp)) != NULL)
  {
    if (std::string(dp->d_name) == "." || std::string(dp->d_name) == "..")
      continue;
    v.push_back(dp->d_name);
  }
  closedir(dirp);
  std::sort(v.begin(),v.end());
}

template<class PointT>   // PointT must contain Normal
void calCloudNormal(boost::shared_ptr<pcl::PointCloud<PointT> > cloud, float radius)
{
  pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
  unsigned NUM_CPU = std::thread::hardware_concurrency();
  ne.setNumberOfThreads(NUM_CPU);
  ne.setInputCloud (cloud);
  boost::shared_ptr<pcl::search::KdTree<PointT> > tree (new pcl::search::KdTree<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

  ne.setSearchMethod (tree);
  ne.setRadiusSearch (radius);
  ne.compute (*cloud_normals);

  pcl::concatenateFields (*cloud, *cloud_normals, *cloud);
}
template void calCloudNormal<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud, float radius);
template void calCloudNormal<pcl::PointSurfel>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud, float radius);

template<class PointT>
void calNormalMLS(boost::shared_ptr<pcl::PointCloud<PointT>> cloud, float normal_radius)
{
  pcl::MovingLeastSquares<PointT, PointT> mls;
  mls.setComputeNormals (true);
  unsigned NUM_CPU = std::thread::hardware_concurrency();
  mls.setNumberOfThreads(NUM_CPU);
  mls.setPolynomialOrder(2);
  mls.setInputCloud (cloud);
  boost::shared_ptr<pcl::search::KdTree<PointT> > tree (new pcl::search::KdTree<PointT>);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (normal_radius);
  boost::shared_ptr<pcl::PointCloud<PointT>> cloud_normal(new pcl::PointCloud<PointT>);
  mls.process (*cloud_normal);
  pcl::PointIndicesPtr indices = mls.getCorrespondingIndices ();
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (indices);
  extract.setNegative (false);
  extract.filter (*cloud);
  pcl::concatenateFields (*cloud, *cloud_normal, *cloud);
}
template void calNormalMLS<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal>> cloud, float normal_radius);
template void calNormalMLS<pcl::PointSurfel>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel>> cloud, float normal_radius);

template <class PointT>
void calNormalIntegralImage(boost::shared_ptr<pcl::PointCloud<PointT>> cloud, int method, float max_depth_change_factor, float smooth_size, bool depth_dependent_smooth)
{
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  ne.setViewPoint(0.0, 0.0, 0.0);
  if (depth_dependent_smooth)
  {
    ne.setDepthDependentSmoothing(true);
  }
  else
  {
    ne.setDepthDependentSmoothing(false);
  }
  switch (method)
  {
    case 0:
      ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
      break;
    case 1:
      ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
      break;
    case 2:
      ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE);  // worst
      break;
    default:
      ne.setNormalEstimationMethod(ne.SIMPLE_3D_GRADIENT );  // better than AVERAGE_DEPTH_CHANGE, speed similar
  }

  ne.setMaxDepthChangeFactor(max_depth_change_factor);
  ne.setNormalSmoothingSize(smooth_size);
  ne.setInputCloud(cloud);
  ne.compute(*normals);

  pcl::concatenateFields (*cloud, *normals, *cloud);
}
template void calNormalIntegralImage<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal>> cloud, int method, float max_depth_change_factor, float smooth_size,bool depth_dependent_smooth);


template<class PointT>
void downsamplePointCloud(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, float vox_size)
{
  pcl::VoxelGrid<PointT> vox;
  vox.setInputCloud(cloud_in);
  vox.setLeafSize(vox_size, vox_size, vox_size);
  vox.filter(*cloud_out);
}
template void downsamplePointCloud<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointXYZRGB>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointXYZ>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointSurfel>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud_out, float vox_size);


//Return argsort indices
template <typename T>
std::vector<int> vectorArgsort(const std::vector<T> &v, bool min_to_max)
{

  // initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  if (min_to_max)
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] < v[i2]; });
  else
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
}
template std::vector<int> vectorArgsort(const std::vector<float> &v, bool min_to_max);
template std::vector<int> vectorArgsort(const std::vector<int> &v, bool min_to_max);


//@scene: need to have normal!
//@model: need to have normal!
//@use_normal: default true
template<class PointT>
float computeLCP(boost::shared_ptr<pcl::PointCloud<PointT> > scene, const std::vector<float> &scene_weights, boost::shared_ptr<pcl::PointCloud<PointT> > model, float dist_thres, float angle_thres, bool use_normal, bool use_dot_score, bool use_reciprocal)
{
  float cp = 0;
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud (model);
  pcl::KdTreeFLANN<PointT> kdtree_scene;
  kdtree_scene.setInputCloud (scene);

  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  const float cos_thres = std::cos(angle_thres/180.0*M_PI);
  for (int i=0;i<scene->points.size();i++)
  {
    PointT pt = scene->points[i];
    if (kdtree.nearestKSearch (pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0  && pointNKNSquaredDistance[0]<dist_thres*dist_thres)
    {
      auto nei = model->points[pointIdxNKNSearch[0]];
      if (!use_normal)
      {
        cp+=scene_weights[i];
      }
      else
      {
        Eigen::Vector3f n1, n2;
        n1<<pt.normal_x,pt.normal_y,pt.normal_z;          // scene
        n2<<nei.normal_x,nei.normal_y,nei.normal_z;     // model

        n1.normalize();
        n2.normalize();
        if (n1.dot(n2)>cos_thres)
        {
          if (!use_dot_score)
            cp += scene_weights[i];
          else
            cp += n1.dot(n2)*(1-std::sqrt(pointNKNSquaredDistance[0])/dist_thres) * scene_weights[i];
        }
      }

      if (use_reciprocal)
      {
        auto pt = nei;  // on model
        if (kdtree_scene.nearestKSearch (pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          auto nei = scene->points[pointIdxNKNSearch[0]]; // on scene
          if (!use_normal)
          {
            cp+=scene_weights[i];
          }
          else
          {
            Eigen::Vector3f n1, n2;
            n1<<pt.normal_x,pt.normal_y,pt.normal_z;
            n2<<nei.normal_x,nei.normal_y,nei.normal_z;
            n1.normalize();
            n2.normalize();
            if (n1.dot(n2)>cos_thres)
            {
              if (!use_dot_score)
                cp += scene_weights[i];
              else
                cp += n1.dot(n2)*(1-std::sqrt(pointNKNSquaredDistance[0])/dist_thres) * scene_weights[i];
            }
          }

        }
      }
    }
  }
  return cp;

}
template float computeLCP(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > scene, const std::vector<float> &scene_weights,boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > model, float dist_thres, float angle_thres, bool use_normal, bool use_dot_score, bool use_reciprocal);
template float computeLCP(boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > scene, const std::vector<float> &scene_weights,boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > model, float dist_thres, float angle_thres, bool use_normal,bool use_dot_score, bool use_reciprocal);
template float computeLCP(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > scene, const std::vector<float> &scene_weights, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > model, float dist_thres, float angle_thres, bool use_normal,bool use_dot_score, bool use_reciprocal);



template<class PointT>
float computeAverageDistance(boost::shared_ptr<pcl::PointCloud<PointT> > scene, boost::shared_ptr<pcl::PointCloud<PointT> > model)
{
  float dist = 0;
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud (model);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  for (int i=0;i<scene->points.size();i++)
  {
    PointT pt = scene->points[i];
    if (kdtree.nearestKSearch (pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    {
      dist += std::sqrt(pointNKNSquaredDistance[0]);
      std::cout<<"std::sqrt(pointNKNSquaredDistance[0]) = "<<std::sqrt(pointNKNSquaredDistance[0])<<"\n";
    }
  }
  dist /= static_cast<int>(scene->points.size());
  return dist;
}
template float computeAverageDistance(boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > scene, boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > model);
template float computeAverageDistance(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > scene, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > model);



template <class PointT>
void removeAllNaNFromPointCloud(boost::shared_ptr<pcl::PointCloud<PointT> > cloud)
{
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  boost::shared_ptr<pcl::PointCloud<PointT> > tmp(new pcl::PointCloud<PointT>);
  tmp->header = cloud->header;
  tmp->points.reserve(cloud->points.size ());
  tmp->sensor_origin_ = cloud->sensor_origin_;
  tmp->sensor_orientation_ = cloud->sensor_orientation_;

  for (int i=0;i<cloud->points.size();i++)
  {
    PointT pt = cloud->points[i];
    if  (std::isfinite(pt.x) && std::isfinite(pt.y) &&std::isfinite(pt.z))
    {
      tmp->push_back(pt);
    }
  }
  tmp->swap(*cloud);
}
template void removeAllNaNFromPointCloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud);
template void removeAllNaNFromPointCloud(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud);

template<class T>
Eigen::Matrix<T,3,3> eulerToRotationMatrix(T roll, T pitch, T yaw)
{
  using namespace Eigen;
  Matrix<T,3,3> R;
  R = AngleAxis<T>(yaw, Eigen::Matrix<T,3,1>::UnitZ())
    * AngleAxis<T>(pitch, Eigen::Matrix<T,3,1>::UnitY())
    * AngleAxis<T>(roll, Eigen::Matrix<T,3,1>::UnitX());
  return R;
}
template Eigen::Matrix<float,3,3> eulerToRotationMatrix(float roll, float pitch, float yaw);
template Eigen::Matrix<double,3,3> eulerToRotationMatrix(double roll, double pitch, double yaw);



void parsePoseTxt(std::string filename, Eigen::Matrix4f &out)
{
  using namespace std;
  std::vector<float> data;
  string line;
  ifstream file(filename);
  if (file.is_open())
  {
    while (getline(file, line))
    { // get a whole line
      std::stringstream ss(line);
      while (getline(ss, line, ' '))
      {
        // You now have separate entites here
        if (line.size()>0)
          data.push_back(stof(line));
      }
    }
  }
  else
  {
    std::cout<<"opening failed: \n"<<filename<<std::endl;
  }
  for (int i=0;i<16;i++)
  {
    out(i/4,i%4) = data[i];
  }
}

} // namespace Utils



ostream& operator<< (ostream& os, const Eigen::Quaternionf &q)
{
  os<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<"\n";
  return os;
}



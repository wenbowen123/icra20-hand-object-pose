#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <camera_constants.h>
#include <simulation_io.hpp>

// For OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace Eigen;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::io;
using namespace pcl::simulation;
using namespace std;

SimExample::Ptr simexample;
pcl::simulation::Scene::Ptr scene_;

static void writeDepthImage(cv::Mat &depthImg, std::string path)
{
  cv::Mat depthImgRaw = cv::Mat::zeros(depthImg.rows, depthImg.cols, CV_16UC1);
  for (int u = 0; u < depthImg.rows; u++)
    for (int v = 0; v < depthImg.cols; v++)
    {
      float depth = depthImg.at<float>(u, v) * 10000;
      unsigned short depthShort = (unsigned short)depth;
      depthShort = (depthShort << 3 | depthShort >> 13);
      depthImgRaw.at<unsigned short>(u, v) = depthShort;
    }
  cv::imwrite(path, depthImgRaw);
}

void clearScene()
{
  scene_->clear();
}

void addObjects(pcl::PolygonMesh::Ptr mesh)
{
  PolygonMeshModel::Ptr transformed_mesh = PolygonMeshModel::Ptr(new PolygonMeshModel(GL_POLYGON, mesh));
  scene_->add(transformed_mesh);
}

/*
  Output: depth_image
*/
void renderDepth(Eigen::Matrix4f pose, cv::Mat &depth_image, std::string path)
{
  Eigen::Isometry3d camera_pose;
  camera_pose.setIdentity();

  Eigen::Vector3d trans;
  Eigen::Matrix3d rot;

  for (int ii = 0; ii < 3; ii++)
    for (int jj = 0; jj < 3; jj++)
      rot(ii, jj) = pose(ii, jj);
  trans << pose(0, 3), pose(1, 3), pose(2, 3);

  camera_pose = camera_pose * rot;
  Matrix3d m;
  m = AngleAxisd(0, Vector3d::UnitZ()) * AngleAxisd(0, Vector3d::UnitY()) * AngleAxisd(M_PI / 2, Vector3d::UnitX());
  camera_pose *= m;
  m = AngleAxisd(M_PI / 2, Vector3d::UnitZ()) * AngleAxisd(0, Vector3d::UnitY()) * AngleAxisd(0, Vector3d::UnitX());
  camera_pose *= m;
  camera_pose.translation() = trans;
  simexample->doSim(camera_pose);

  const float *depth_buffer = simexample->rl_->getDepthBuffer();
  simexample->get_depth_image_cv(depth_buffer, depth_image);
  depth_image.convertTo(depth_image, CV_32FC1);
  depth_image = depth_image / 1000;
  depth_image.setTo(0, depth_image > 1);
  // writeDepthImage(depth_image, path);
}

void initScene(int argc, char **argv)
{
  int width = kCameraWidth;
  int height = kCameraHeight;
  simexample = SimExample::Ptr(new SimExample(argc, argv, height, width, kCameraFX, kCameraFY, kCameraCX, kCameraCY));

  scene_ = simexample->scene_;
  if (scene_ == NULL)
  {
    printf("ERROR: Scene is not set\n");
  }
}

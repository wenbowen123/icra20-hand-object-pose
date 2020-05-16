#include <iostream>
#include <camera.h>
#include <camera_constants.h>

using namespace Eigen;
using namespace pcl::simulation;

// Initialize to some default values, but allow users to change elsewhere.
// Parameters for MS Kinect V1.0
// int kCameraWidth = 640;
// int kCameraHeight = 480;
// float kCameraFX = 576.09757860f;
// float kCameraFY = 576.09757860f;
// float kCameraCX = 321.06398107f;
// float kCameraCY = 242.97676897f;
// float kZNear = 0.1f;
// float kZFar = 20.0f;

// Parameters for MS Kinect V2.0
// APC: these values are from calib_ir.yaml
// int kCameraWidth = 512;
// int kCameraHeight = 424;
// float kCameraFX = 354.76029132272464f;
// float kCameraFY = 354.49153515383603f;
// float kCameraCX = 253.06754087288363f;
// float kCameraCY = 205.37746288838397f;
// float kZNear = 0.1f;
// float kZFar = 20.0f;

// For RealSense
int kCameraWidth = 640;
int kCameraHeight = 480;
float kCameraFX = 616.5961303710938;
float kCameraFY = 616.59619140625;
float kCameraCX = 307.6278076171875;
float kCameraCY = 239.68692016601562;
float kZNear = 0.1f;
float kZFar = 2.0f;

void
pcl::simulation::Camera::move (double vx, double vy, double vz)
{
  Vector3d v;
  v << vx, vy, vz;
  pose_.pretranslate (pose_.rotation ()*v);
  x_ = pose_.translation ().x ();
  y_ = pose_.translation ().y ();
  z_ = pose_.translation ().z ();
}

void
pcl::simulation::Camera::updatePose ()
{
  Matrix3d m;
  m = AngleAxisd (yaw_, Vector3d::UnitZ ())
    * AngleAxisd (pitch_, Vector3d::UnitY ())
    * AngleAxisd (roll_, Vector3d::UnitX ());

  pose_.setIdentity ();
  pose_ *= m;
  
  Vector3d v;
  v << x_, y_, z_;
  pose_.translation () = v;
}

void
pcl::simulation::Camera::setParameters (int width, int height,
                                        float fx, float fy,
                                        float cx, float cy,
                                        float z_near, float z_far)
{
  width_ = width;
  height_ = height;
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  z_near_ = z_near;
  z_far_ = z_far;

  float z_nf = (z_near_-z_far_);
  projection_matrix_ <<  2.0f*fx_/width_,  0,                 1.0f-(2.0f*cx_/width_),     0,
                         0,                2.0f*fy_/height_,  1.0f-(2.0f*cy_/height_),    0,
                         0,                0,                (z_far_+z_near_)/z_nf,  2.0f*z_near_*z_far_/z_nf,
                         0,                0,                -1.0f,                  0;
}

void
pcl::simulation::Camera::initializeCameraParameters ()
{
  setParameters (kCameraWidth, kCameraHeight,
                 kCameraFX, kCameraFY,
                 kCameraCX, kCameraCY,
                 kZNear, kZFar); //ZNEAR
}

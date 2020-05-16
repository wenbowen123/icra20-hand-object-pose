#include "Renderer.h"


using namespace Eigen;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::io;
using namespace pcl::simulation;


Renderer::Renderer(int H, int W, float fx, float fy, float cx, float cy):HEIGHT(H),WIDTH(W)
{
  _simexample = SimExample::Ptr(new SimExample(0, NULL, HEIGHT, WIDTH, fx,fy,cx,cy));

  if (_simexample->scene_ == NULL)
  {
    printf("ERROR: Scene is not set\n");
  }
}


Renderer::~Renderer()
{

}

void Renderer::reset()
{
  clearScene();
  _simexample->reset();
}


void Renderer::clearScene()
{
  _simexample->scene_->clear();
}


//Mesh format is strict. Use Meshlab to get the mesh. If the mesh has no color, output rgb will be red
void Renderer::addObject(pcl::PolygonMesh::Ptr mesh, const Eigen::Matrix4f &pose)
{
  pcl::PolygonMesh::Ptr mesh1(new pcl::PolygonMesh);
  Utils::transformPolygonMesh(mesh, mesh1, pose);
  PolygonMeshModel::Ptr transformed_mesh = PolygonMeshModel::Ptr(new PolygonMeshModel(GL_POLYGON, mesh1));
  _simexample->scene_->add(transformed_mesh);
}

//@pose: cvcam in world frame. If world frame and cam frame is same, then this is Identity. Camera axis follow CV style
//@depth_image: output, unit in meter
void Renderer::doRender(const Eigen::Matrix4d &pose, cv::Mat &depth_image, cv::Mat &bgr)
{
  Eigen::Vector3d trans = pose.block(0,3,3,1);
  Eigen::Matrix3d rot = pose.block(0,0,3,3);


  Eigen::Matrix4d glcam_in_cvcam;
  glcam_in_cvcam.setIdentity();

  //Turn to Z-up, X-forward coordinate cam frame
  glcam_in_cvcam.block(0,0,3,3)<<0, -1, 0,
                                 0, 0, -1,
                                 1, 0, 0;

  Eigen::Matrix4d glcam_in_ob = pose * glcam_in_cvcam;
  Eigen::Isometry3d camera_pose;
  camera_pose.matrix() = glcam_in_ob;


  _simexample->doSim(camera_pose);
  const float *depth_buffer = _simexample->rl_->getDepthBuffer();
  _simexample->get_depth_image_cv(depth_buffer, depth_image);
  depth_image.convertTo(depth_image, CV_32FC1);
  depth_image = depth_image / 1000.0;
  depth_image.setTo(2.0, depth_image > 2.0);
  depth_image.setTo(0.1, depth_image < 0.1);

  const uint8_t *rgb_buffer = _simexample->rl_->getColorBuffer ();
  _simexample->get_rgb_image_cv(rgb_buffer,bgr);

}

void Renderer::removeLastObject()
{
  _simexample->scene_->removeLast();
}


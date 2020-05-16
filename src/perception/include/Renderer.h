#ifndef RENDERER_HHHH__
#define RENDERER_HHHH__


#include "Utils.h"
#include <boost/shared_ptr.hpp>
#include <camera_constants.h>
#include <simulation_io.hpp>

class Renderer
{
public:
  Renderer(int H, int W, float fx, float fy, float cx, float cy);
  ~Renderer();
  void reset();
  void clearScene();
  void addObject(pcl::PolygonMesh::Ptr mesh, const Eigen::Matrix4f &pose);
  void doRender(const Eigen::Matrix4d &pose, cv::Mat &depth_image, cv::Mat &bgr);
  void removeLastObject();


public:
  pcl::simulation::SimExample::Ptr _simexample;
  int HEIGHT,WIDTH;

};



#endif
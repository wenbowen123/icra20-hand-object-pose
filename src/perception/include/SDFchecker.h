#ifndef SDF_CHECKER_HH__
#define SDF_CHECKER_HH__

#include "Utils.h"
#include "igl/readOBJ.h"
#include "igl/writeOBJ.h"
#include "igl/signed_distance.h"

class SDFchecker
{
public:
  SDFchecker();
  ~SDFchecker();
  void reset();
  void transformVertices(Eigen::MatrixXf &V, const Eigen::Matrix4f &pose);
  void registerMesh(std::string mesh_dir, std::string mesh_name, const Eigen::Matrix4f &pose);
  void registerMesh(pcl::PolygonMesh::Ptr mesh, std::string mesh_name, const Eigen::Matrix4f &pose);
  void getSignedDistanceMinMax(const Eigen::MatrixXf &pts, float lower_bound, float upper_bound, float &min_dist, float &max_dist, std::vector<float> &dists, int &num_inner_pt);
  void getSignedDistanceMinMaxWithRegistered(const std::string &registered_name, const Eigen::MatrixXf &pts, float lower_bound, float upper_bound, float &min_dist, float &max_dist, std::vector<float> &dists, int &num_inner_pt);
  void transformMesh(std::string mesh_name, const Eigen::Matrix4f &pose);
  void saveVerticesToPLY(std::string name, std::string out_name);
  void saveMeshObj(std::string name, std::string out_dir);


public:
  std::map<std::string, Eigen::MatrixXf> _Vertices;
  std::map<std::string, Eigen::MatrixXi> _Faces;

  
};


#endif
#include "SDFchecker.h"


SDFchecker::SDFchecker()
{

}

SDFchecker::~SDFchecker()
{

}

void SDFchecker::reset()
{
  _Vertices.clear();
  _Faces.clear();
}

//@V: Nx3 vertices
void SDFchecker::transformVertices(Eigen::MatrixXf &V, const Eigen::Matrix4f &pose)
{
  assert(V.cols()==3);
  V.transposeInPlace();
  Eigen::MatrixXf ones=Eigen::MatrixXf::Ones(1,V.cols());
  Eigen::MatrixXf V_homo(V.rows()+ones.rows(),V.cols());
  V_homo<<V,ones;

  V_homo = pose * V_homo;
  V=V_homo.block(0,0,3,V_homo.cols());
  V.transposeInPlace();
}


void SDFchecker::registerMesh(std::string mesh_dir, std::string mesh_name, const Eigen::Matrix4f &pose)
{
  Eigen::MatrixXf V1;
  Eigen::MatrixXi F;
  igl::readOBJ(mesh_dir, V1,F);
  Eigen::MatrixXf V=V1.block(0,0,V1.rows(),3);

  transformVertices(V,pose);

  _Vertices[mesh_name]=V;
  _Faces[mesh_name]=F;


}

void SDFchecker::registerMesh(pcl::PolygonMesh::Ptr mesh, std::string mesh_name, const Eigen::Matrix4f &pose)
{

  Eigen::MatrixXi F;
  PointCloud::Ptr cloud(new PointCloud);
  pcl::fromPCLPointCloud2(mesh->cloud, *cloud);
  Eigen::MatrixXf cloud_eigen = cloud->getMatrixXfMap();
  Eigen::MatrixXf V = cloud_eigen.block(0,0,3,cloud_eigen.cols());
  V.transposeInPlace();
  assert(V.cols()==3);
  F.resize(mesh->polygons.size(),3);

  #pragma omp parallel for
  for (int i=0;i<mesh->polygons.size();i++)
  {
    assert(mesh->polygons[i].vertices.size()==3); //Has to be trimesh
    for (int j=0;j<3;j++)
    {
      F(i,j) = mesh->polygons[i].vertices[j];
    }
  }
  transformVertices(V,pose);

  _Vertices[mesh_name]=V;
  _Faces[mesh_name]=F;


}


void SDFchecker::transformMesh(std::string mesh_name, const Eigen::Matrix4f &pose)
{
  Eigen::MatrixXf P=_Vertices[mesh_name];
  transformVertices(P, pose);
  _Vertices[mesh_name]=P;

}

//@pts: Nx3
void SDFchecker::getSignedDistanceMinMax(const Eigen::MatrixXf &pts, float lower_bound, float upper_bound, float &min_dist, float &max_dist, std::vector<float> &dists, int &num_inner_pt)
{
  assert(pts.cols()==3);
  igl::SignedDistanceType sign_type = igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
  min_dist=std::numeric_limits<float>::max();
  max_dist=-std::numeric_limits<float>::max();
  num_inner_pt=0;
  dists.clear();
  for (auto h:_Vertices)
  {
    Eigen::VectorXf S,I;
    Eigen::MatrixXf C,N;
    std::string name=h.first;
    Eigen::MatrixXf V=h.second;
    Eigen::MatrixXi F=_Faces[name];
    igl::signed_distance(pts,V,F,sign_type,lower_bound, upper_bound, S,I,C,N);
    std::vector<float> tmp(S.data(), S.data() + S.rows());
    dists.insert(dists.begin(), tmp.begin(), tmp.end());

    min_dist=std::min(min_dist, S.minCoeff());
    max_dist=std::max(max_dist, S.maxCoeff());

  }

}

void SDFchecker::getSignedDistanceMinMaxWithRegistered(const std::string &registered_name, const Eigen::MatrixXf &pts, float lower_bound, float upper_bound, float &min_dist, float &max_dist, std::vector<float> &dists, int &num_inner_pt)
{
  assert(pts.cols()==3);
  igl::SignedDistanceType sign_type = igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
  min_dist=std::numeric_limits<float>::max();
  max_dist=-std::numeric_limits<float>::max();
  num_inner_pt=0;
  dists.clear();

  Eigen::VectorXf S,I;
  Eigen::MatrixXf C,N;
  Eigen::MatrixXf V=_Vertices[registered_name];
  Eigen::MatrixXi F=_Faces[registered_name];
  igl::signed_distance(pts,V,F,sign_type,lower_bound, upper_bound, S,I,C,N);
  dists.insert(dists.begin(), S.data(), S.data() + S.rows());

  min_dist=S.minCoeff();
  max_dist=S.maxCoeff();

}

void SDFchecker::saveVerticesToPLY(std::string name, std::string out_name)
{
  Eigen::MatrixXf V=_Vertices[name];
  PointCloud::Ptr pts(new PointCloud);
  for (int i=0;i<V.rows();i++)
  {
    pcl::PointXYZ pt;
    pt.x=V(i,0);
    pt.y=V(i,1);
    pt.z=V(i,2);
    pts->points.push_back(pt);
  }
  pcl::io::savePLYFile("/home/bowen/debug/"+out_name+".ply", *pts);
}

void SDFchecker::saveMeshObj(std::string name, std::string out_dir)
{
  Eigen::MatrixXf V=_Vertices[name];
  Eigen::MatrixXi F=_Faces[name];
  igl::writeOBJ(out_dir, V, F);
}
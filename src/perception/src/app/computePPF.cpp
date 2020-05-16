#include "Utils.h"
#include "ConfigParser.h"

int ppfClosestBin(int value, int discretization) {

  int lower_limit = value - (value % discretization);
  int upper_limit = lower_limit + discretization;

  int dist_from_lower = value - lower_limit;
  int dist_from_upper = upper_limit - value;

  int closest = (dist_from_lower < dist_from_upper)? lower_limit:upper_limit;

  return closest;
}

void computePPF(pcl::PointXYZRGBNormal pt1, pcl::PointXYZRGBNormal pt2, std::vector<int> &ppf)
{
  const float DIST_DISCRET = 5;  // mm
  const float ANGLE_DISCRET = 10;  //degree
  ppf.clear();
  ppf.resize(4);
  Eigen::Vector3f n1(pt1.normal_x, pt1.normal_y, pt1.normal_z);
  Eigen::Vector3f n2(pt2.normal_x, pt2.normal_y, pt2.normal_z);
  n1.normalize();
  n2.normalize();
  Eigen::Vector3f p1(pt1.x, pt1.y, pt1.z);
  Eigen::Vector3f p2(pt2.x, pt2.y, pt2.z);
  int dist = static_cast<int>((p1-p2).norm()*1000);
  Eigen::Vector3f p1p2 = p2-p1;
  int n1_p1p2 = static_cast<int>(std::acos(n1.dot(p1p2.normalized())) / M_PI * 180);
  int n2_p1p2 = static_cast<int>(std::acos(n2.dot(p1p2.normalized())) / M_PI * 180);
  int n1_n2 = static_cast<int>(std::acos(n1.dot(n2)) / M_PI * 180);
  ppf[0] = ppfClosestBin(dist, DIST_DISCRET);
  ppf[1] = ppfClosestBin(n1_p1p2, ANGLE_DISCRET);
  ppf[2] = ppfClosestBin(n2_p1p2, ANGLE_DISCRET);
  ppf[3] = ppfClosestBin(n1_n2, ANGLE_DISCRET);
}

int main(int argc, char **argv)
{
  if (argc<4)
  {
    std::cout<<"Arguments:\narg1: config_dir, arg2: out_dir"<<std::endl;
    return 1;
  }
  ros::init(argc,argv,"computePPF");
  const std::string config_dir = std::string(argv[1]);
  const std::string out_dir = std::string(argv[2]);

  ConfigParser cfg(config_dir);

  PointCloudRGBNormal::Ptr cloud(new PointCloudRGBNormal);
  pcl::io::loadPLYFile(cfg.object_model_path, *cloud);

  float downsample_size = 0.001;
  float normal_radius = 0.003;
  float ppf_density = 0.005;

  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(cloud, cloud, downsample_size);
  pcl::PointXYZRGBNormal minPt, maxPt;
  pcl::getMinMax3D (*cloud, minPt, maxPt);
  pcl::PointXYZRGBNormal mid_pt;
  mid_pt.x = (minPt.x+maxPt.x)/2.0;
  mid_pt.y = (minPt.y+maxPt.y)/2.0;
  mid_pt.z = (minPt.z+maxPt.z)/2.0;
  for (auto &pt:cloud->points)
  {
    pt.x = pt.x - mid_pt.x;
    pt.y = pt.y - mid_pt.y;
    pt.z = pt.z - mid_pt.z;
  }

  Utils::calNormalMLS<pcl::PointXYZRGBNormal>(cloud, normal_radius);
  for (auto &pt:cloud->points)
  {
    pcl::flipNormalTowardsViewpoint (pt, 0, 0, 0,
          pt.normal[0],
          pt.normal[1],
          pt.normal[2]);
    pt.normal[0] = -pt.normal[0];
    pt.normal[1] = -pt.normal[1];
    pt.normal[2] = -pt.normal[2];
  }
  pcl::io::savePLYFile(out_dir+"/model.ply", *cloud);
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(cloud, cloud, ppf_density);
  std::map<std::vector<int>, std::vector<std::pair<int,int> > > ppfs;
  for (int i=0;i<cloud->points.size();i++)
  {
    for (int j=i+1;j<cloud->points.size();j++)
    {
      std::vector<int> ppf;
      computePPF(cloud->points[i], cloud->points[j], ppf);
      ppfs[ppf].push_back(std::make_pair(i,j));
      ppfs[ppf].push_back(std::make_pair(j,i));
    }
  }

  for (auto &pt:cloud->points)
  {
    pt.x = pt.x + mid_pt.x;
    pt.y = pt.y + mid_pt.y;
    pt.z = pt.z + mid_pt.z;
  }
  std::ofstream f(out_dir+"ppf", std::ios::binary);
  boost::archive::binary_oarchive oa(f);
  oa << ppfs;
}
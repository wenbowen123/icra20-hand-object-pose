#include "Hand.h"
#include "pugixml.hpp"
#include "yaml-cpp/yaml.h"
#include "optim.hpp"
#include "SDFchecker.h"

using namespace Eigen;


double objFuncPSO(const arma::vec& X, arma::vec* grad_out, optim::ArgPasser* args)
{
  const std::string name = args->name;
  const float dist_thres = args->dist_thres;
  float score = 0;
  Eigen::Matrix4f tf_self;
  {
    Eigen::Matrix3f R;  // Finger rotate around x
    R = AngleAxisf(0, Vector3f::UnitZ()) * AngleAxisf(0, Vector3f::UnitY()) * AngleAxisf(X[0], Vector3f::UnitX());
    tf_self.setIdentity();
    tf_self.block(0,0,3,3) = R;
  }
  Eigen::Matrix4f cur_model2handbase = args->model2handbase * tf_self;

  //We penalize when a pair of gripper too close to hold anything, NOTE: y is inner side. We consider both sides of the finger, tip1:tip side, tip2:palm side
  Eigen::Vector4f tip1, tip2;
  if (name=="finger_1_1" || name=="finger_2_1")   //Palm side finger
  {
    tip1 << args->finger_out_property._min_x, args->finger_out_property._max_y, args->finger_out_property._min_z, 1;
    Eigen::Matrix4f cur_finger_out2handbase = args->model2handbase * tf_self * args->finger_out2parent;
    tip1 = cur_finger_out2handbase*tip1;

    tip2 << args->finger_property._min_x, args->finger_property._max_y, args->finger_property._min_z, 1;
    tip2 = cur_model2handbase*tip2;
  }
  else
  {
    tip1 << args->finger_property._min_x, args->finger_property._max_y, args->finger_property._min_z, 1;
    tip1 = cur_model2handbase*tip1;

    tip2 << args->finger_property._min_x, args->finger_property._max_y, args->finger_property._max_z, 1;
    tip2 = cur_model2handbase*tip2;
  }

  float gripper_dist1, gripper_dist2;
  if (name=="finger_2_1" || name=="finger_2_2")   //Right side finger
  {
    gripper_dist1 = tip1(1)-args->pair_tip1(1);
    gripper_dist2 = tip2(1)-args->pair_tip2(1);
  }
  else
  {
    gripper_dist1 = -tip1(1)+args->pair_tip1(1);
    gripper_dist2 = -tip2(1)+args->pair_tip2(1);
  }


  float penalty_gripper_dist = 0;
  const float GRIPPER_MIN_DIST = args->cfg.gripper_min_dist;
  if (gripper_dist1<GRIPPER_MIN_DIST || gripper_dist2<GRIPPER_MIN_DIST)
  {
    penalty_gripper_dist = 1e3 + 1e3*std::abs(GRIPPER_MIN_DIST-gripper_dist1);  //NOTE: only use dist1 to make objective monotone
    score -= penalty_gripper_dist;
    return -score;   //NOTE: we are minimizing the cost function
  }


  float num_match = 0;
  const float PLANAR_DIST_THRES = args->cfg.yml["hand_match"]["planar_dist_thres"].as<float>();
  PointCloudRGBNormal::Ptr model = args->model;
  //NOTE: we compare in handbase frame, so that kdtree only build once
  PointCloudRGBNormal::Ptr model_in_handbase(new PointCloudRGBNormal);
  pcl::transformPointCloudWithNormals(*model, *model_in_handbase, cur_model2handbase);
  float NORMAL_ANGLE_THRES = -1;
  if (name=="finger_1_1" || name=="finger_2_1")
  {
    NORMAL_ANGLE_THRES = std::cos(args->cfg.yml["hand_match"]["finger1_normal_angle"].as<float>()/180.0*M_PI);
  }
  else
  {
    NORMAL_ANGLE_THRES = std::cos(args->cfg.yml["hand_match"]["finger2_normal_angle"].as<float>()/180.0*M_PI);
  }
  // Eigen::MatrixXf V(model_in_handbase->points.size(),3);
  const bool check_normal = args->cfg.yml["hand_match"]["check_normal"].as<bool>();
  for (int ii=0;ii<model_in_handbase->points.size();ii++)
  {
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    auto pt = model_in_handbase->points[ii];
    if (args->kdtree_scene->nearestKSearch (pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    {
      auto nei = args->scene_hand_region->points[pointIdxNKNSearch[0]];
      float sq_planar_dist = (pt.x-nei.x)*(pt.x-nei.x) + (pt.y-nei.y)*(pt.y-nei.y);  //In handbase's frame

      if (pointNKNSquaredDistance[0]<=dist_thres*dist_thres /*|| (sq_planar_dist<=PLANAR_DIST_THRES*PLANAR_DIST_THRES && std::abs(pt.z-nei.z)<=0.03)*/)  // Squared dist!!
      {
        if (!check_normal)
        {
          // num_match+=std::exp(-sq_planar_dist/(PLANAR_DIST_THRES*PLANAR_DIST_THRES));
          num_match += 1 + X[0];
          continue;
        }
        if (nei.normal_x==0 && nei.normal_y==0 && nei.normal_z==0)
        {
          num_match += 1 + X[0];
          continue;
        }
        if (std::isfinite(nei.normal_x) && std::isfinite(nei.normal_y) && std::isfinite(nei.normal_z))
        {
          // V.block(num_match,0,1,3) << nei.x, nei.y, nei.z;
          Eigen::Vector3f n1(pt.normal_x, pt.normal_y, pt.normal_z);
          Eigen::Vector3f n2(nei.normal_x, nei.normal_y, nei.normal_z);
          if (n1.dot(n2)>=NORMAL_ANGLE_THRES)
          {
            // num_match+=n1.dot(n2)*std::exp(-sq_planar_dist/(PLANAR_DIST_THRES*PLANAR_DIST_THRES));
            // num_match += (1-std::sqrt(sq_planar_dist)/dist_thres) + X[0];
            // num_match+=n1.dot(n2);
            num_match += 1 + X[0];
          }
          continue;
        }

      }
    }
  }
  score += num_match;


  //We penalize points on outer side of finger, this only makes sense when it's somewhat matching. Otherwise scene in finger is not aligned with finger property along z axis
  float outer_dist_sum=0;
  int num_outer = 0;
  FingerProperty finger_property = args->finger_property;
  PointCloudRGBNormal::Ptr scene_no_swivel_in_finger(new PointCloudRGBNormal);
  pcl::transformPointCloudWithNormals(*args->scene_remove_swivel, *scene_no_swivel_in_finger, cur_model2handbase.inverse());

  if (num_match==0)    // encourage moving
  {
    score = -100 + X[0];
    return -score;
  }

  for (const auto &pt:scene_no_swivel_in_finger->points)
  {
    int cur_bin = finger_property.getBinAlongZ(pt.z);
    if (pt.y>=finger_property._hist_alongz(1,cur_bin))  //Same axis for all fingers
    {
      continue;
    }
    // if (pt.y>=-0.01) continue;
    // outer_dist_sum += std::abs(pt.y+0.01);
    outer_dist_sum += std::abs(pt.y-finger_property._hist_alongz(1,cur_bin));
    num_outer++;
  }
  float penalty_outer=0;
  const int MAX_OUTER_PTS = args->cfg.yml["hand_match"]["max_outter_pts"].as<int>();
  const float outter_pt_dist = args->cfg.yml["hand_match"]["outter_pt_dist"].as<float>();
  const float outter_pt_dist_weight = args->cfg.yml["hand_match"]["outter_pt_dist_weight"].as<float>();
  // if ( (num_outer>=20 && outer_dist_sum/num_outer-outter_pt_dist>0) || num_outer>=MAX_OUTER_PTS )
  // {
  //   penalty_outer = 1e3 + 1e3*std::max(outer_dist_sum/num_outer-outter_pt_dist, 0.0f);
  //   score -= penalty_outer;
  // }
  float avg_outer_dist = outer_dist_sum/num_outer;
  if (num_outer>=MAX_OUTER_PTS || avg_outer_dist>=0.005)
  {
    penalty_outer = 1e3 + outter_pt_dist_weight*std::max(avg_outer_dist-outter_pt_dist, 0.0f);
    score -= penalty_outer;
  }
  else if ( (num_outer>=0 && avg_outer_dist-outter_pt_dist>0))
  {
    // penalty_outer = outter_pt_dist_weight*(avg_outer_dist-outter_pt_dist);
    penalty_outer = outter_pt_dist_weight*std::exp(avg_outer_dist*1000);
    score -= penalty_outer;
  }

  return -score;   //NOTE: we are minimizing the cost function


}

FingerProperty::FingerProperty(){};

FingerProperty::FingerProperty(PointCloudRGBNormal::Ptr model, int num_division):_num_division(num_division)
{
  pcl::PointXYZRGBNormal min_pt, max_pt;
  pcl::getMinMax3D(*model, min_pt, max_pt);
  _min_x = min_pt.x;
  _min_y = min_pt.y;
  _min_z = min_pt.z;
  _max_x = max_pt.x;
  _max_y = max_pt.y;
  _max_z = max_pt.z;
  _stride_z = (_max_z-_min_z)/num_division;
  _hist_alongz.resize(6,num_division);
  _hist_alongz.block(0,0,3,num_division).setConstant(std::numeric_limits<float>::max());
  _hist_alongz.block(3,0,3,num_division).setConstant(-std::numeric_limits<float>::max());
  std::vector<bool> changed(num_division, false);
  for (int i=0;i<model->points.size();i++)
  {
    auto pt = model->points[i];
    int bin = getBinAlongZ(pt.z);
    _hist_alongz(0,bin) = std::min(_hist_alongz(0,bin), pt.x);
    _hist_alongz(1,bin) = std::min(_hist_alongz(1,bin), pt.y);
    _hist_alongz(2,bin) = std::min(_hist_alongz(2,bin), pt.z);
    _hist_alongz(3,bin) = std::max(_hist_alongz(3,bin), pt.x);
    _hist_alongz(4,bin) = std::max(_hist_alongz(4,bin), pt.y);
    _hist_alongz(5,bin) = std::max(_hist_alongz(5,bin), pt.z);
    changed[bin]=true;
  }

  //When cur bin never touched, take nearest valid neighbor
  for (int i=0;i<num_division;i++)
  {
    if (changed[i]) continue;
    for (int j=i+1;j<num_division;j++)
    {
      if (changed[j])
      {
        _hist_alongz.col(i) = _hist_alongz.col(j);
        changed[i] = true;
        break;
      }
    }
  }
  if (changed[num_division-1]==false)
  {
    for (int i=num_division-2;i>=0;i--)
    {
      if (changed[i])
      {
        _hist_alongz.col(num_division-1) = _hist_alongz.col(i);
        changed[num_division-1] = true;
        break;
      }
    }
  }
}

FingerProperty::~FingerProperty()
{

}


int FingerProperty::getBinAlongZ(float z)
{
  int bin = std::max(z-_min_z, 0.0f)/_stride_z;
  bin = std::max(bin,0);
  bin = std::min(bin, _num_division-1);
  return bin;
}


Hand::Hand()
{

}

Hand::Hand(ConfigParser *cfg1, const Eigen::Matrix3f &cam_K)
{
  cfg=cfg1;
  _hand_cloud = boost::make_shared<PointCloudRGBNormal>();
  _cam_K = cam_K;

  parseURDF();
  for (auto h:_clouds)
  {
    std::string name = h.first;
    if (name.find("finger")==-1) continue;
    _finger_properties[name] = FingerProperty(h.second, 10);
  }
  initPSO();

}

Hand::~Hand()
{
}

void Hand::setCurScene(const cv::Mat &depth_meters, PointCloudRGBNormal::Ptr scene_organized, PointCloudRGBNormal::Ptr scene_hand_region, const Eigen::Matrix4f &handbase_in_cam)
{
  _depth_meters=depth_meters;
  _scene_sampled=boost::make_shared<PointCloudRGBNormal>();
  _scene_hand_region=boost::make_shared<PointCloudRGBNormal>();
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(scene_organized, _scene_sampled, 0.001);
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(scene_hand_region, _scene_hand_region,0.003);
  _handbase_in_cam = handbase_in_cam;
  handbaseICP(scene_organized);

  //NOTE: we compare in handbase frame, so that kdtree only build once
  PointCloudRGBNormal::Ptr scene_in_handbase(new PointCloudRGBNormal);
  pcl::transformPointCloudWithNormals(*_scene_hand_region, *scene_in_handbase, _handbase_in_cam.inverse());
  PointCloudRGBNormal::Ptr scene_hand_region_removed_noise(new PointCloudRGBNormal);
  {
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> outrem;
    outrem.setInputCloud(scene_in_handbase);
    outrem.setRadiusSearch(0.02);
    outrem.setMinNeighborsInRadius (30);
    outrem.filter (*scene_hand_region_removed_noise);
  }
  {
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> outrem;
    outrem.setInputCloud(scene_hand_region_removed_noise);
    outrem.setRadiusSearch(0.04);
    outrem.setMinNeighborsInRadius (100);
    outrem.filter (*scene_hand_region_removed_noise);
  }
  {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
    sor.setInputCloud (scene_hand_region_removed_noise);
    sor.setMeanK (20);
    sor.setStddevMulThresh (2);
    sor.filter (*scene_hand_region_removed_noise);   //! in hand base
  }
  PointCloudRGBNormal::Ptr scene_remove_swivel(new PointCloudRGBNormal);
  {
    pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
    pass.setInputCloud (scene_hand_region_removed_noise);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (-0.25, -0.1);
    pass.filter (*scene_remove_swivel);   // In handbase frame
  }
  assert(scene_remove_swivel->points.size()>0);
  PointCloudRGBNormal::Ptr scene_noswivel_cam(new PointCloudRGBNormal);
  pcl::transformPointCloudWithNormals(*scene_remove_swivel, *scene_noswivel_cam, _handbase_in_cam);
  PointCloudRGBNormal::Ptr handregion_in_cam(new PointCloudRGBNormal);
  pcl::transformPointCloudWithNormals(*scene_hand_region_removed_noise, *handregion_in_cam, _handbase_in_cam);
  _pso_args.scene_hand_region_removed_noise = scene_hand_region_removed_noise;
  _pso_args.scene_hand_region = scene_in_handbase;
  boost::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> > kdtree(new pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>);
  kdtree->setInputCloud(scene_hand_region_removed_noise);
  _pso_args.kdtree_scene = kdtree;
  _pso_args.scene_remove_swivel = scene_remove_swivel;

}

void Hand::reset()
{
  _kdtrees.clear();
  _matched_scene_indices.clear();
  for (auto &c:_component_status)
  {
    c.second = false;
  }
  for (auto &h:_finger_angles)
  {
    h.second = 0;
  }
  _handbase_in_cam.setIdentity();
  _hand_cloud->clear();
  _depth_meters.release();
  _scene_sampled->clear();
  _scene_hand_region->clear();
  _pso_args.reset();
  for (auto &t:_tf_self)
  {
    t.second.setIdentity();
  }

}

void Hand::printComponents()
{
  std::cout<<"\nprint hand info:\n";
  for (auto h:_clouds)
  {
    std::string name=h.first;
    std::string parent_name=_parent_names[name];
    std::cout<<"name="<<name<<", parent="<<parent_name<<"\n";
    std::cout<<"tf_self:\n"<<_tf_self[name]<<"\n";
    std::cout<<"tf_in_parent:\n"<<_tf_in_parent[name]<<"\n";
  }
  std::cout<<"\n";
}

void Hand::parseURDF()
{
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file((cfg->yml["urdf_path"].as<std::string>()).c_str());
  assert(result);

  for (pugi::xml_node tool = doc.child("robot").child("link"); tool; tool = tool.next_sibling("link"))
  {
    auto visual=tool.child("visual");
    std::string name=tool.attribute("name").value();

    if (name.find("rail")!=-1) continue;

    std::vector<float> rpy(3);
    if (!visual.child("origin").attribute("rpy"))
    {
      rpy.resize(3,0);
    }
    else
    {
      std::string rpy_str=visual.child("origin").attribute("rpy").value();
      Utils::delimitString(rpy_str, ' ', rpy);
    }


    std::vector<float> xyz(3);
    if (!visual.child("origin").attribute("xyz"))
    {
      xyz.resize(3,0);
    }
    else
    {
      std::string xyz_str=visual.child("origin").attribute("xyz").value();
      Utils::delimitString(xyz_str,' ', xyz);
    }

    Eigen::Matrix3f R;
    R=AngleAxisf(rpy[2], Vector3f::UnitZ()) * AngleAxisf(rpy[1], Vector3f::UnitY()) * AngleAxisf(rpy[0], Vector3f::UnitX());

    Eigen::Matrix4f tf_init(Eigen::Matrix4f::Identity());
    tf_init.block(0,0,3,3)=R;
    tf_init.block(0,3,3,1)=Eigen::Vector3f(xyz[0],xyz[1],xyz[2]);

    Eigen::Matrix4f tf_in_parent(Eigen::Matrix4f::Identity());
    std::string parent_name;
    for (pugi::xml_node tool = doc.child("robot").child("joint"); tool; tool = tool.next_sibling("joint"))
    {
      if (tool.child("child").attribute("link").value()!=name) continue;
      parent_name=tool.child("parent").attribute("link").value();

      std::string rpy_str=tool.child("origin").attribute("rpy").value();
      std::vector<float> rpy(3,0);
      Utils::delimitString(rpy_str, ' ', rpy);
      std::string xyz_str=tool.child("origin").attribute("xyz").value();
      std::vector<float> xyz(3,0);
      Utils::delimitString(xyz_str, ' ', xyz);

      Eigen::Matrix3f R;
      R=AngleAxisf(rpy[2], Vector3f::UnitZ()) * AngleAxisf(rpy[1], Vector3f::UnitY()) * AngleAxisf(rpy[0], Vector3f::UnitX());

      tf_in_parent.block(0,0,3,3)=R;
      tf_in_parent.block(0,3,3,1)=Eigen::Vector3f(xyz[0],xyz[1],xyz[2]);
      break;
    }


    std::vector<float> scale(3,1);
    if (!visual.child("geometry").child("mesh").attribute("scale"))
    {
      scale.resize(3,1);
    }
    else
    {
      std::string scale_str=visual.child("geometry").child("mesh").attribute("scale").value();
      Utils::delimitString(scale_str, ' ', scale);
    }


    PointCloudRGBNormal::Ptr cloud(new PointCloudRGBNormal);
    pcl::io::loadPLYFile(cfg->yml["Hand"][name]["cloud"].as<std::string>(), *cloud);
    assert(cloud->points.size()>0);
    for (auto &pt:cloud->points)
    {
      pt.x *= scale[0];
      pt.y *= scale[1];
      pt.z *= scale[2];
    }



    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
    pcl::io::loadOBJFile(cfg->yml["Hand"][name]["mesh"].as<std::string>(), *mesh);
    PointCloud::Ptr mesh_cloud(new PointCloud);
    pcl::fromPCLPointCloud2(mesh->cloud, *mesh_cloud);
    assert(mesh_cloud->points.size()>0);
    for (auto &pt:mesh_cloud->points)
    {
      pt.x *= scale[0];
      pt.y *= scale[1];
      pt.z *= scale[2];
    }
    pcl::toPCLPointCloud2(*mesh_cloud, mesh->cloud);

    // Component init pose must be applied at beginning according to URDF !!
    pcl::PolygonMesh::Ptr convex_mesh(new pcl::PolygonMesh);
    pcl::io::loadOBJFile(cfg->yml["Hand"][name]["convex_mesh"].as<std::string>(), *convex_mesh);
    PointCloud::Ptr convex_mesh_cloud(new PointCloud);
    pcl::fromPCLPointCloud2(convex_mesh->cloud, *convex_mesh_cloud);
    assert(convex_mesh_cloud->points.size()>0);
    for (auto &pt:convex_mesh_cloud->points)
    {
      pt.x *= scale[0];
      pt.y *= scale[1];
      pt.z *= scale[2];
    }
    pcl::toPCLPointCloud2(*convex_mesh_cloud, convex_mesh->cloud);

    // Component init pose must be applied at beginning according to URDF !!
    pcl::transformPointCloudWithNormals(*cloud,*cloud,tf_init);
    Utils::transformPolygonMesh(mesh,mesh,tf_init);
    Utils::transformPolygonMesh(convex_mesh,convex_mesh,tf_init);


    std::cout<<"adding component name:"+name<<std::endl;
    addComponent(name,parent_name,cloud,mesh,convex_mesh,tf_in_parent,Eigen::Matrix4f::Identity());
  }

}

//Component tf in handbase, accounting for self rotation
void Hand::getTFHandBase(std::string cur_name, Eigen::Matrix4f &tf_in_handbase)
{
  tf_in_handbase.setIdentity();
  while (1)   //Get pose in world
  {
    if (cur_name=="base_link")
    {
      break;
    }
    if (_tf_self.find(cur_name)==_tf_self.end())
    {
      std::cout<<"cur_name does not exist!!!\n";
      exit(1);
    }
    Matrix4f cur_tf = _tf_in_parent[cur_name];
    tf_in_handbase = cur_tf * _tf_self[cur_name] * tf_in_handbase;
    cur_name = _parent_names[cur_name];
  }
}

void Hand::addComponent(std::string name, std::string parent_name, PointCloudRGBNormal::Ptr cloud, pcl::PolygonMesh::Ptr mesh, pcl::PolygonMesh::Ptr convex_mesh, const Matrix4f &tf_in_parent, const Matrix4f &tf_self)
{
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(cloud, cloud, 0.005);
  _clouds[name] = cloud;
  _meshes[name] = mesh;
  _convex_meshes[name] = convex_mesh;
  _parent_names[name] = parent_name;
  _tf_in_parent[name] = tf_in_parent;
  _tf_self[name] = tf_self;
}

//Return cloud in handbase frame
void Hand::makeHandCloud()
{
  _hand_cloud->clear();
  for (auto h:_clouds)
  {
    std::string name = h.first;
    PointCloudRGBNormal::Ptr component_cloud = h.second;

    Eigen::Matrix4f model2handbase;
    getTFHandBase(name,model2handbase);
    PointCloudRGBNormal::Ptr tmp(new PointCloudRGBNormal);
    pcl::transformPointCloudWithNormals(*component_cloud, *tmp, model2handbase);

    (*_hand_cloud) += (*tmp);

    boost::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> > kdtree(new pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>);
    kdtree->setInputCloud(tmp);
    _kdtrees[name] = kdtree;
  }
}

//For visualization
void Hand::makeHandMesh()
{
  for (auto h:_meshes)
  {
    std::string name = h.first;
    std::cout<<"making hand mesh "+name<<std::endl;
    pcl::PolygonMesh::Ptr cur_mesh = h.second;
    Eigen::Matrix4f model2handbase;
    getTFHandBase(name,model2handbase);
    pcl::PolygonMesh::Ptr tmp(new pcl::PolygonMesh);
    Utils::transformPolygonMesh(cur_mesh, tmp, model2handbase);
    Utils::transformPolygonMesh(tmp, tmp, _handbase_in_cam);
  }
  for (auto h:_convex_meshes)
  {
    std::string name = h.first;
    pcl::PolygonMesh::Ptr cur_mesh = h.second;
    Eigen::Matrix4f model2handbase;
    getTFHandBase(name,model2handbase);
    pcl::PolygonMesh::Ptr tmp(new pcl::PolygonMesh);
    Utils::transformPolygonMesh(cur_mesh, tmp, model2handbase);
    pcl::io::saveOBJFile("/home/bowen/debug/handmesh/convex_"+name+".obj", *tmp);
    Utils::transformPolygonMesh(tmp, tmp, _handbase_in_cam);
    pcl::io::saveOBJFile("/home/bowen/debug/handmesh/convex_incam_"+name+".obj", *tmp);
  }
}


void Hand::initPSO()
{
  _pso_args.cfg = *cfg;
  _pso_settings.pso_n_pop = cfg->yml["hand_match"]["pso"]["n_pop"].as<int>();
  _pso_settings.pso_n_gen = cfg->yml["hand_match"]["pso"]["n_gen"].as<int>();
  _pso_settings.vals_bound = true;
  _pso_settings.pso_check_freq = cfg->yml["hand_match"]["pso"]["check_freq"].as<int>();
  _pso_settings.err_tol = 1e-5;
  _pso_settings.pso_par_c_cog = cfg->yml["hand_match"]["pso"]["pso_par_c_cog"].as<float>();
  _pso_settings.pso_par_c_soc = cfg->yml["hand_match"]["pso"]["pso_par_c_soc"].as<float>();
  _pso_settings.pso_par_initial_w = cfg->yml["hand_match"]["pso"]["pso_par_initial_w"].as<float>();


}


bool Hand::matchOneComponentPSO(std::string model_name, float min_angle, float max_angle, bool use_normal, float dist_thres, float normal_angle_thres, float least_match)
{
  _pso_args.dist_thres = dist_thres;
  _pso_settings.upper_bounds = arma::zeros(1) + max_angle*M_PI/180;
  _pso_settings.lower_bounds = arma::zeros(1) + min_angle*M_PI/180;
  _pso_settings.pso_initial_ub = arma::zeros(1) + max_angle*M_PI/180;
  _pso_settings.pso_initial_lb = arma::zeros(1) + min_angle*M_PI/180;

  std::map<std::string, std::string> pair_names;
  pair_names["finger_1_1"] = "finger_2_1";
  pair_names["finger_2_1"] = "finger_1_1";
  pair_names["finger_1_2"] = "finger_2_2";
  pair_names["finger_2_2"] = "finger_1_2";
  Eigen::Matrix4f pair_in_base;
  const std::string pair_name = pair_names[model_name];
  if (pair_name=="finger_1_1" || pair_name=="finger_2_1")  // This case, We consider the whole staight finger1+2
  {
    const std::string pair_finger_out_name = pair_name=="finger_1_1"? "finger_1_2" : "finger_2_2";
    Eigen::Vector4f pair_tip_pt1(_finger_properties[pair_finger_out_name]._min_x, _finger_properties[pair_finger_out_name]._max_y, _finger_properties[pair_finger_out_name]._min_z, 1);
    getTFHandBase(pair_finger_out_name, pair_in_base);
    pair_tip_pt1 = pair_in_base * pair_tip_pt1;
    _pso_args.pair_tip1 = pair_tip_pt1;

    getTFHandBase(pair_name, pair_in_base);
    Eigen::Vector4f pair_tip_pt2(_finger_properties[pair_name]._min_x, _finger_properties[pair_name]._max_y, _finger_properties[pair_name]._min_z, 1);
    pair_tip_pt2 = pair_in_base * pair_tip_pt2;
    _pso_args.pair_tip2 = pair_tip_pt2;
    const std::string finger_out_name = model_name=="finger_1_1"? "finger_1_2" : "finger_2_2";
    _pso_args.finger_out2parent = _tf_in_parent[finger_out_name];
    _pso_args.finger_out_property = _finger_properties[finger_out_name];
  }
  else
  {
    Eigen::Vector4f pair_tip_pt1(_finger_properties[pair_name]._min_x, _finger_properties[pair_name]._max_y, _finger_properties[pair_name]._min_z, 1);
    getTFHandBase(pair_name, pair_in_base);
    pair_tip_pt1 = pair_in_base * pair_tip_pt1;
    _pso_args.pair_tip1 = pair_tip_pt1;
    Eigen::Vector4f pair_tip_pt2(_finger_properties[pair_name]._min_x, _finger_properties[pair_name]._max_y, _finger_properties[pair_name]._max_z, 1);
    pair_tip_pt2 = pair_in_base * pair_tip_pt2;
    _pso_args.pair_tip2 = pair_tip_pt2;
  }

  arma::vec X = arma::zeros(1);
  _pso_args.name = model_name;
  _pso_args.model = _clouds[model_name];
  Eigen::Matrix4f model2handbase;
  getTFHandBase(model_name, model2handbase);
  _pso_args.model2handbase = model2handbase;
  Eigen::Matrix4f model_in_cam = _handbase_in_cam * model2handbase;
  _pso_args.model_in_cam = model_in_cam;
  _pso_args.finger_property = _finger_properties[model_name];
  // _pso_args.sdf.registerMesh(_meshes[model_name], model_name, Eigen::Matrix4f::Identity());

  bool success = optim::pso(X,objFuncPSO,&_pso_args,_pso_settings);
  if (!success || -_pso_args.objval<=least_match)
  {
    std::cout<<model_name+" PSO matching failed"<<std::endl;
    _tf_self[model_name].setIdentity();
    _component_status[model_name]=false;
    return false;
  }
  float angle = static_cast<float>(X[0]);
  Eigen::Matrix3f R;
  R = AngleAxisf(0, Vector3f::UnitZ()) * AngleAxisf(0, Vector3f::UnitY()) * AngleAxisf(angle, Vector3f::UnitX());
  _tf_self[model_name].setIdentity();
  _tf_self[model_name].block(0,0,3,3) = R;
  _component_status[model_name]=true;
  printf("%s PSO final angle=%f, match_score=%f\n", model_name.c_str(), angle, -_pso_args.objval);
  return true;
}




void Hand::handbaseICP(PointCloudRGBNormal::Ptr scene_organized)
{
  PointCloudRGBNormal::Ptr scene_sampled(new PointCloudRGBNormal);
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(scene_organized, scene_sampled, 0.005);

  PointCloudRGBNormal::Ptr handbase(new PointCloudRGBNormal);
  pcl::copyPointCloud(*_clouds["base_link"],*handbase);

  PointCloudRGBNormal::Ptr scene_handbase(new PointCloudRGBNormal);
  Eigen::Matrix4f cam_in_handbase = _handbase_in_cam.inverse();  // cam -> handbase^ -> handbase(real) -> finger
  assert(cam_in_handbase!=Eigen::Matrix4f::Identity());
  pcl::transformPointCloudWithNormals(*scene_sampled, *scene_handbase, cam_in_handbase);

  pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
  pass.setInputCloud (scene_handbase);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (-0.07, 0.03);
  pass.filter (*scene_handbase);

  pass.setInputCloud (scene_handbase);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (-0.18, 0.01);
  pass.filter (*scene_handbase);


  float z1=_tf_in_parent["finger_1_1"](2,3);
  float y1=_tf_in_parent["finger_1_1"](1,3);
  float z2=_tf_in_parent["finger_2_1"](2,3);
  float y2=_tf_in_parent["finger_2_1"](1,3);


  // Remove swivel part from handbase
  PointCloudRGBNormal::Ptr scene_handbase_tmp(new PointCloudRGBNormal);
  scene_handbase_tmp->points.reserve(scene_handbase->points.size());

  for (const auto &pt:scene_handbase->points)   // Remove finger connection part
  {
    float sq_dist1 = (pt.z-z1)*(pt.z-z1) + (pt.y-y1)*(pt.y-y1);
    if (sq_dist1<=0.015*0.015) continue;
    float sq_dist2 = (pt.z-z2)*(pt.z-z2) + (pt.y-y2)*(pt.y-y2);
    if (sq_dist2<=0.015*0.015) continue;
    if ((pt.y>=y1 && pt.y<=y2) || (pt.y>=y2 && pt.y<=y1))
    {
      if (std::abs(pt.z-z1)<=0.01 || std::abs(pt.z-z1)<=0.01)
      {
        continue;
      }
    }

    scene_handbase_tmp->points.push_back(pt);


  }
  scene_handbase_tmp->swap(*scene_handbase);

  //Now all in handbase
  Eigen::Matrix4f cam2handbase_offset(Eigen::Matrix4f::Identity());
  float score = Utils::runICP<pcl::PointXYZRGBNormal>(scene_handbase, handbase, cam2handbase_offset, 50, 30, 0.03, 1e-4);

  pcl::transformPointCloudWithNormals(*scene_handbase,*scene_handbase,cam2handbase_offset);

  std::cout<<"cam2handbase_offset:\n"<<cam2handbase_offset<<"\n\n";
  float translation = cam2handbase_offset.block(0,3,3,1).norm();
  if (translation >=0.05)
  {
    printf("cam2handbase_offset set to Identity, icp=%f, translation=%f, x=%f, y=%f\n",score, translation, std::abs(cam2handbase_offset(0,3)), std::abs(cam2handbase_offset(1,3)));
    cam2handbase_offset.setIdentity();
  }

  float rot_diff = Utils::rotationGeodesicDistance(Eigen::Matrix3f::Identity(), cam2handbase_offset.block(0,0,3,3)) / M_PI *180.0;
  Eigen::Matrix3f R = cam2handbase_offset.block(0,0,3,3);   // rotate rpy around static axis
  Eigen::Vector3f rpy = R.eulerAngles(2,1,0);
  float pitch = rpy(1); // Rotation along y axis
  pitch = std::min(std::abs(pitch), std::abs(static_cast<float>(M_PI)-pitch));
  pitch = std::min(std::abs(pitch), std::abs(static_cast<float>(M_PI)+pitch));
  if (rot_diff>=10 || std::abs(pitch)>=10/180.0*M_PI)
  {
    cam2handbase_offset.setIdentity();
    printf("cam2handbase_offset set to Identity");
  }
  if (cam2handbase_offset!=Eigen::Matrix4f::Identity())
  {
    _component_status["handbase"] = true;
  }
  _handbase_in_cam = _handbase_in_cam*cam2handbase_offset.inverse();

}

HandT42::HandT42(ConfigParser *cfg1, const Eigen::Matrix3f &cam_K):Hand(cfg1, cam_K)
{
}

HandT42::HandT42()
{

}

HandT42::~HandT42()
{
}


//Operations in handbase frame. Hack: alternative to SDF with similar result while slightly faster.
//@dist_thres: distance square !!
template<class PointT, bool has_normal>
void HandT42::removeSurroundingPointsAndAssignProbability(boost::shared_ptr<pcl::PointCloud<PointT> > scene, boost::shared_ptr<pcl::PointCloud<PointT> > scene_out, float dist_thres)
{
  scene_out->clear();
  scene_out->points.reserve(scene->points.size());
  pcl::transformPointCloudWithNormals(*scene, *scene, _handbase_in_cam.inverse());
  scene_out->points.reserve(scene->points.size());
  const float lambda = 231.04906018664843; // Exponential distribution: this makes about 0.003m the prob is 0.5
#pragma omp parallel
  {
    std::map<std::string, boost::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>>> kdtrees_local;
    for (auto &h : _kdtrees)
    {
      kdtrees_local[h.first].reset(new pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>(*h.second));
    }

    #pragma omp for schedule(dynamic)
    for (int i = 0; i < scene->points.size(); i++)
    {
      auto pt = scene->points[i];
      pcl::PointXYZRGBNormal pt_tmp;
      pt_tmp.x = pt.x;
      pt_tmp.y = pt.y;
      pt_tmp.z = pt.z;

      bool is_near = false;
      float min_dist = 1.0;
      for (auto h : kdtrees_local)
      {
        float local_dist_thres = dist_thres;
        std::string name = h.first;
        if (name == "finger_2_1" || name == "finger_1_1")
        {
          local_dist_thres = 0.005 * 0.005;
        }
        else if (name == "base" || name == "swivel_1" || name == "swivel_2")
        {
          local_dist_thres = 0.02 * 0.02;
        }

        boost::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>> kdtree = h.second;
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        if (kdtree->nearestKSearch(pt_tmp, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          min_dist = std::min(min_dist, std::sqrt(pointNKNSquaredDistance[0]));
          if (pointNKNSquaredDistance[0] <= local_dist_thres)
          {
            is_near = true;
            break;
          }
          auto nei = kdtree->getInputCloud()->points[pointIdxNKNSearch[0]];
          float sq_planar_dist = (pt_tmp.x - nei.x) * (pt_tmp.x - nei.x) + (pt_tmp.y - nei.y) * (pt_tmp.y - nei.y);
          if (sq_planar_dist <= local_dist_thres && std::abs(pt_tmp.z - nei.z) <= 0.005)
          {
            is_near = true;
            break;
          }
        }
      }

      if (!is_near)
      {
        pt.confidence = 1 - std::exp(-lambda * min_dist);
        #pragma omp critical
        scene_out->points.push_back(pt);
      }
    }
  }

  //Remove outter side points
  boost::shared_ptr<pcl::PointCloud<PointT> > scene_out_in_finger_2_2(new pcl::PointCloud<PointT>);
  boost::shared_ptr<pcl::PointCloud<PointT> > scene_out_in_finger_1_2(new pcl::PointCloud<PointT>);
  {
    Eigen::Matrix4f finger2_2_in_handbase;
    getTFHandBase("finger_2_2",finger2_2_in_handbase);
    if (has_normal)
      pcl::transformPointCloudWithNormals(*scene_out,*scene_out_in_finger_2_2,finger2_2_in_handbase.inverse());
    else
      pcl::transformPointCloud(*scene_out,*scene_out_in_finger_2_2,finger2_2_in_handbase.inverse());
  }

  {
    Eigen::Matrix4f finger1_2_in_handbase;
    getTFHandBase("finger_1_2",finger1_2_in_handbase);
    if (has_normal)
      pcl::transformPointCloudWithNormals(*scene_out,*scene_out_in_finger_1_2,finger1_2_in_handbase.inverse());
    else
      pcl::transformPointCloud(*scene_out,*scene_out_in_finger_1_2,finger1_2_in_handbase.inverse());
  }

  const float min_z = _finger_properties["finger_1_2"]._min_z;
  boost::shared_ptr<pcl::PointCloud<PointT> > tmp(new pcl::PointCloud<PointT>);
  tmp->points.reserve(scene_out->points.size());
  for (int i=0;i<scene_out_in_finger_1_2->points.size();i++)
  {
    auto pt1 = scene_out_in_finger_1_2->points[i];
    if (pt1.y<0 && pt1.z>=min_z) continue;

    auto pt2 = scene_out_in_finger_2_2->points[i];
    if (pt2.y<0 && pt2.z>=min_z) continue;

    auto pt = scene_out->points[i];
    tmp->points.push_back(pt);
  }
  pcl::transformPointCloudWithNormals(*tmp, *scene_out, _handbase_in_cam);
}
template void HandT42::removeSurroundingPointsAndAssignProbability<pcl::PointSurfel,true>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > scene, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > scene_out, float dist_thres);


void HandT42::fingerICP()
{

  int num_near_finger_2_1=0, num_near_finger_1_1=0;
  PointCloudRGBNormal::Ptr finger_2_1(new PointCloudRGBNormal);
  PointCloudRGBNormal::Ptr finger_1_1(new PointCloudRGBNormal);

  // we operate in hand base frame
  {
    Eigen::Matrix4f model2handbase(Eigen::Matrix4f::Identity());
    getTFHandBase("finger_2_1",model2handbase);
    pcl::transformPointCloudWithNormals(*_clouds["finger_2_1"], *finger_2_1, model2handbase);
  }
  {
    Eigen::Matrix4f model2handbase(Eigen::Matrix4f::Identity());
    getTFHandBase("finger_1_1",model2handbase);
    pcl::transformPointCloudWithNormals(*_clouds["finger_1_1"], *finger_1_1, model2handbase);
  }

  PointCloudRGBNormal::Ptr scene_handbase(new PointCloudRGBNormal);
  Eigen::Matrix4f cam_in_handbase = _handbase_in_cam.inverse();  // cam -> handbase^ -> handbase(real) -> finger
  pcl::transformPointCloudWithNormals(*_scene_sampled, *scene_handbase, cam_in_handbase);
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(scene_handbase, scene_handbase, 0.005);

  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree1, kdtree2;
  kdtree1.setInputCloud (finger_1_1);
  kdtree2.setInputCloud (finger_2_1);

  PointCloudRGBNormal::Ptr scene_icp(new PointCloudRGBNormal);
  for (const auto &pt:scene_handbase->points)
  {
    {
      std::vector<int> pointIdxNKNSearch(1);
      std::vector<float> pointNKNSquaredDistance(1);
      if (kdtree1.nearestKSearch (pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
      {
        if (pointNKNSquaredDistance[0]<=0.03*0.03)
        {
          auto nei = finger_1_1->points[pointIdxNKNSearch[0]];
          float sq_dist_planar = (nei.x-pt.x)*(nei.x-pt.x) + (nei.y-pt.y)*(nei.y-pt.y);
          if (sq_dist_planar<0.01*0.01)
          {
            num_near_finger_1_1++;
            scene_icp->points.push_back(pt);
            continue;
          }
        }
      }
    }

    {
      std::vector<int> pointIdxNKNSearch(1);
      std::vector<float> pointNKNSquaredDistance(1);
      if (kdtree2.nearestKSearch (pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
      {
        if (pointNKNSquaredDistance[0]<=0.03*0.03)
        {
          auto nei = finger_2_1->points[pointIdxNKNSearch[0]];
          float sq_dist_planar = (nei.x-pt.x)*(nei.x-pt.x) + (nei.y-pt.y)*(nei.y-pt.y);
          if (sq_dist_planar<0.01*0.01)
          {
            num_near_finger_2_1++;
            scene_icp->points.push_back(pt);
            continue;
          }
        }
      }
    }

  }

  pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
  pass.setInputCloud (scene_icp);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (-1000.0, 0.0);
  pass.filter (*scene_icp);

  pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> outrem;
  outrem.setInputCloud(scene_icp);
  outrem.setRadiusSearch(0.01);
  outrem.setMinNeighborsInRadius (10);
  outrem.filter (*scene_icp);


  PointCloudRGBNormal::Ptr finger_icp(new PointCloudRGBNormal);
  if (num_near_finger_1_1>10)
  {
    *finger_icp += *finger_1_1;
  }
  if (num_near_finger_2_1>10)
  {
    *finger_icp += *finger_2_1;
  }


  //Now all in handbase
  Eigen::Matrix4f cam2handbase_offset(Eigen::Matrix4f::Identity());
  Utils::runICP<pcl::PointXYZRGBNormal>(scene_icp, finger_icp, cam2handbase_offset, 50, 15, 0.02);

  pcl::transformPointCloudWithNormals(*scene_icp, *scene_icp, cam2handbase_offset);

  _handbase_in_cam = _handbase_in_cam*cam2handbase_offset.inverse();
}



//NOTE: Only do this when handbase match fail
void HandT42::adjustHandHeight()
{
  makeHandCloud();
  if ( _component_status["handbase"]==true )
  {
    return;
  }
  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
  PointCloudRGBNormal::Ptr scene_handbase(new PointCloudRGBNormal);  //Compare in handbase frame
  pcl::transformPointCloudWithNormals(*_scene_hand_region, *scene_handbase, _handbase_in_cam.inverse());
  kdtree.setInputCloud(scene_handbase);
  std::vector<float> trial_heights = {-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03};
  int max_match = 0;
  float best_height = 0;
  Eigen::Matrix4f best_offset(Eigen::Matrix4f::Identity());
  makeHandCloud();
  for (int i=0;i<trial_heights.size();i++)
  {
    /*
         T^handbase_cam                         offset
    cam------------------> handbase_hat <--------------------- true handbase
    */
    Eigen::Matrix4f offset(Eigen::Matrix4f::Identity());
    offset(2,3) = trial_heights[i];
    PointCloudRGBNormal::Ptr cur_hand(new PointCloudRGBNormal);
    pcl::transformPointCloudWithNormals(*_hand_cloud, *cur_hand, offset);
    int cur_match = 0;
    for  (const auto &pt:cur_hand->points)
    {
      std::vector<int> indices;
      std::vector<float> sq_dists;
      if (kdtree.nearestKSearch (pt, 1, indices, sq_dists) > 0)
      {
        if (sq_dists[0] > 0.005*0.005) continue;
        auto nei = scene_handbase->points[indices[0]];
        Eigen::Vector3f pt_normal(pt.normal[0], pt.normal[1], pt.normal[2]);
        Eigen::Vector3f nei_normal(nei.normal[0], nei.normal[1], nei.normal[2]);
        if (pt_normal.dot(nei_normal) >= std::cos(45 / 180.0 * M_PI))
        {
          cur_match++;
        }
      }
    }
    if (cur_match>max_match)
    {
      max_match = cur_match;
      best_height = trial_heights[i];
      best_offset = offset;
    }

  }
  _handbase_in_cam = _handbase_in_cam*best_offset;
}

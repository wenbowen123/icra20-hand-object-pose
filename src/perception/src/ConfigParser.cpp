#include "ConfigParser.h"


void parsePoseTxt(std::string filename, std::vector<float> &data)
{
  using namespace std;
  data.clear();
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
}


ConfigParser::ConfigParser(std::string cfg_file)
{
  yml = YAML::LoadFile(cfg_file);
  parseYMLFile(cfg_file);
}

ConfigParser::~ConfigParser()
{

}


void ConfigParser::parseYMLFile(std::string filepath)
{
	system(("rosparam load " + filepath).c_str());
  {
    std::vector<float> K;
    nh.getParam("cam_K", K);
    for (int i=0;i<9;i++)
    {
      cam_intrinsic(i/3,i%3) = K[i];
    }
  }

  endeffector2global.setIdentity();
  Eigen::Matrix3f R_tmp;
  R_tmp<<0,1,0,
         -1,0,0,
         0,0,1;
  endeffector2global.block(0,0,3,3) = R_tmp;

  nh.getParam("cam_in_world", cam_in_world_file);
  std::vector<float> cam_in_world_data;
  parsePoseTxt(cam_in_world_file,cam_in_world_data);
  if (cam_in_world_data.size()==0)
  {
    exit(1);
  }
  for (int i=0;i<16;i++)
  {
    cam_in_world(i/4,i%4) = cam_in_world_data[i];
  }
  std::cout<<"cam_in_world:\n"<<cam_in_world<<"\n\n";

  {
    std::vector<float> data;
    nh.getParam("cam1_in_leftarm", data);
    cam1_in_leftarm.setIdentity();
    cam1_in_leftarm.block(0,3,3,1)<<data[0],data[1],data[2];
    Eigen::Quaternionf q(data[6],data[3],data[4],data[5]);
    cam1_in_leftarm.block(0,0,3,3) = q.normalized().toRotationMatrix();
  }
  std::cout<<"cam1_in_leftarm:\n"<<cam1_in_leftarm<<"\n\n";

  {
    std::vector<float> data;
    std::string file_dir;
    nh.getParam("palm_in_baselink", file_dir);
    parsePoseTxt(file_dir,data);
    for (int i=0;i<16;i++)
    {
      palm_in_baselink(i/4,i%4) = data[i];
    }
  }
  std::cout<<"palm_in_baselink:\n"<<palm_in_baselink<<"\n\n";

  {
    std::vector<float> data;
    std::string file_dir;
    nh.getParam("leftarm_in_base", file_dir);
    parsePoseTxt(file_dir,data);
    for (int i=0;i<16;i++)
    {
      leftarm_in_base(i/4,i%4) = data[i];
    }
  }

  {
    std::vector<float> data;
    nh.getParam("handbase_in_palm", data);
    for (int i=0;i<16;i++)
    {
      handbase_in_palm(i/4,i%4) = data[i];
    }
  }
  std::cout<<"handbase_in_palm:\n"<<handbase_in_palm<<"\n\n";

  rgb_path = yml["rgb_path"].as<std::string>();
  depth_path = yml["depth_path"].as<std::string>();
  object_model_path = yml["object_model_path"].as<std::string>();
  object_mesh_path = yml["object_mesh_path"].as<std::string>();


  //================== pcl options ========================
  leaf_size = yml["down_sample"]["leaf_size"].as<float>();
  radius = yml["remove_noise"]["radius"].as<float>();
  min_number = yml["remove_noise"]["min_number"].as<float>();
  super4pcs_sample_size = yml["super4pcs_sample_size"].as<float>();
  super4pcs_overlap = yml["super4pcs_overlap"].as<float>();
  super4pcs_delta = yml["super4pcs_delta"].as<float>();
  super4pcs_max_normal_difference = yml["super4pcs_max_normal_difference"].as<float>();
  super4pcs_max_color_distance = yml["super4pcs_max_color_distance"].as<float>();
  super4pcs_max_time_seconds = yml["super4pcs_max_time_seconds"].as<float>();
  pose_estimator_wrong_ratio = yml["pose_estimator_wrong_ratio"].as<float>();
  pose_estimator_high_confidence_thres = yml["pose_estimator_high_confidence_thres"].as<float>();


}

#include "PoseHypo.h"


PoseHypo::PoseHypo()
{
  _id=-1;
  _pose.setIdentity();
  _wrong_ratio=1;
  _lcp_score=0;
}

PoseHypo::PoseHypo(const int &id)
{
  _id=id;
  _pose.setIdentity();
  _wrong_ratio=1;
  _lcp_score=0;
}

PoseHypo::PoseHypo(const Eigen::Matrix4f &pose, const int &id)
{
  _id=id;
  _pose=pose;
  _wrong_ratio=1;
  _lcp_score=0;
}

PoseHypo::PoseHypo(const Eigen::Matrix4f &pose, const int &id, const float &lcp_score)
{
  _id=id;
  _pose=pose;
  _lcp_score=lcp_score;
  _wrong_ratio=1;
}


PoseHypo::~PoseHypo()
{

}



void PoseHypo::print()
{
  std::cout<<"pose#"<<_id<<", lcp_score="<<_lcp_score<<", wrong_ratio="<<_wrong_ratio<<std::endl;
}
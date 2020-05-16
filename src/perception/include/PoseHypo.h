#ifndef POSE_HYPO_HH__
#define POSE_HYPO_HH__

#include "Utils.h"


class PoseHypo
{
public:
  PoseHypo();
  PoseHypo(const int &id);
  PoseHypo(const Eigen::Matrix4f &pose, const int &id);
  PoseHypo(const Eigen::Matrix4f &pose, const int &id, const float &lcp_score);

  ~PoseHypo();
  void print();

  

public:
  Eigen::Matrix4f _pose;
  float _wrong_ratio;
  float _lcp_score;
  int _id;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


#endif
/*
 * scene.cpp
 *
 *  Created on: Aug 16, 2011
 *      Author: Hordur Johannsson
 */

#include <scene.h>

namespace pcl
{

namespace simulation
{

void
Scene::add (Model::Ptr model)
{
  models_.push_back(model);
}

void Scene::removeLast()
{
  models_.pop_back();
}

void
Scene::addCompleteModel (std::vector<Model::Ptr> model)
{
  for (const auto &m:model)
  {
    models_.push_back(m);
  }
}

void
Scene::draw ()
{
  for (std::vector<Model::Ptr>::iterator model = models_.begin (); model != models_.end (); ++model)
    (*model)->draw ();
}

void
Scene::clear ()
{
  models_.clear();
}

} // namespace - simulation
} // namespace - pcl

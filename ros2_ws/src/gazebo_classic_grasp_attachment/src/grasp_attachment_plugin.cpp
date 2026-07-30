#include <algorithm>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace gazebo_classic_grasp_attachment
{
class GraspAttachmentPlugin : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = std::move(world);
    ros_node_ = gazebo_ros::Node::Get(sdf);
    parent_model_name_ = Read<std::string>(sdf, "parent_model", "el_a3");
    parent_link_name_ = Read<std::string>(sdf, "parent_link", "grasp_center");
    max_distance_m_ = Read<double>(sdf, "max_distance_m", 0.060);
    for (auto item = sdf->GetElement("target_model"); item;
         item = item->GetNextElement("target_model")) {
      target_models_.push_back(item->Get<std::string>());
    }
    if (target_models_.empty()) {
      target_models_ = {"red_cube", "blue_cube"};
    }
    service_ = ros_node_->create_service<std_srvs::srv::SetBool>(
      "/grasp_attachment/set_enabled",
      std::bind(&GraspAttachmentPlugin::Request, this,
                std::placeholders::_1, std::placeholders::_2));
    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&GraspAttachmentPlugin::Update, this));
    RCLCPP_INFO(
      ros_node_->get_logger(),
      "grasp attachment ready: parent=%s::%s, max_distance=%.3f m",
      parent_model_name_.c_str(), parent_link_name_.c_str(), max_distance_m_);
  }

private:
  template<typename T>
  T Read(const sdf::ElementPtr & sdf, const std::string & name, const T & fallback) const
  {
    return sdf->HasElement(name) ? sdf->Get<T>(name) : fallback;
  }

  void Request(const std_srvs::srv::SetBool::Request::SharedPtr request,
               std_srvs::srv::SetBool::Response::SharedPtr response)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    requested_enabled_ = request->data;
    has_request_ = true;
    // The physics update performs the actual CreateJoint/Detach safely.
    response->success = true;
    response->message = request->data ? "attach request accepted" : "detach request accepted";
  }

  void Update()
  {
    bool enabled = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!has_request_) {
        return;
      }
      enabled = requested_enabled_;
      has_request_ = false;
    }
    if (enabled) {
      AttachNearestTarget();
    } else {
      Detach();
    }
  }

  void AttachNearestTarget()
  {
    if (joint_) {
      return;
    }
    const auto parent_model = world_->ModelByName(parent_model_name_);
    const auto parent_link = parent_model ? parent_model->GetLink(parent_link_name_) : nullptr;
    if (!parent_link) {
      RCLCPP_ERROR(ros_node_->get_logger(), "attachment parent link %s::%s is unavailable",
                   parent_model_name_.c_str(), parent_link_name_.c_str());
      return;
    }
    gazebo::physics::LinkPtr closest;
    double closest_distance = std::numeric_limits<double>::infinity();
    for (const auto & model_name : target_models_) {
      const auto model = world_->ModelByName(model_name);
      const auto link = model ? model->GetLink("link") : nullptr;
      if (!link) {
        continue;
      }
      const double distance = (parent_link->WorldPose().Pos() - link->WorldPose().Pos()).Length();
      if (distance < closest_distance) {
        closest = link;
        closest_distance = distance;
      }
    }
    if (!closest || closest_distance > max_distance_m_) {
      RCLCPP_WARN(ros_node_->get_logger(),
                  "attachment skipped: nearest cube distance %.3f m exceeds %.3f m",
                  closest_distance, max_distance_m_);
      return;
    }
    joint_ = world_->Physics()->CreateJoint("fixed", parent_model);
    joint_->Load(parent_link, closest, ignition::math::Pose3d());
    joint_->Init();
    attached_target_ = closest->GetModel()->GetName();
    RCLCPP_INFO(ros_node_->get_logger(), "attached %s at %.3f m", attached_target_.c_str(), closest_distance);
  }

  void Detach()
  {
    if (!joint_) {
      return;
    }
    joint_->Detach();
    joint_.reset();
    RCLCPP_INFO(ros_node_->get_logger(), "detached %s", attached_target_.c_str());
    attached_target_.clear();
  }

  gazebo::physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr ros_node_;
  gazebo::event::ConnectionPtr update_connection_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_;
  gazebo::physics::JointPtr joint_;
  std::mutex mutex_;
  bool requested_enabled_{false};
  bool has_request_{false};
  std::string parent_model_name_;
  std::string parent_link_name_;
  std::string attached_target_;
  std::vector<std::string> target_models_;
  double max_distance_m_{0.060};
};

GZ_REGISTER_WORLD_PLUGIN(GraspAttachmentPlugin)
}  // namespace gazebo_classic_grasp_attachment

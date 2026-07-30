#include <algorithm>
#include <functional>
#include <cmath>
#include <string>

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo_classic_gripper_mimic
{
class GripperMimicPlugin : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = std::move(model);
    driver_ = model_->GetJoint(Read<std::string>(sdf, "driver_joint", "L7_joint"));
    left_ = model_->GetJoint(Read<std::string>(sdf, "left_joint", "left_jaw_joint"));
    right_ = model_->GetJoint(Read<std::string>(sdf, "right_joint", "right_jaw_joint"));
    multiplier_ = Read<double>(sdf, "multiplier", -0.031831);
    offset_ = Read<double>(sdf, "offset", 0.05);
    lower_limit_ = Read<double>(sdf, "lower_limit", 0.0);
    upper_limit_ = Read<double>(sdf, "upper_limit", 0.05);
    position_kp_ = Read<double>(sdf, "position_kp", 300.0);
    velocity_kd_ = Read<double>(sdf, "velocity_kd", 2.0);
    max_force_ = Read<double>(sdf, "max_force", 8.0);

    if (!driver_ || !left_ || !right_) {
      gzerr << "gazebo_classic_gripper_mimic: missing driver or jaw joint\n";
      return;
    }
    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&GripperMimicPlugin::UpdateJaws, this));
  }

private:
  template<typename T>
  T Read(const sdf::ElementPtr & sdf, const std::string & name, const T & fallback) const
  {
    return sdf->HasElement(name) ? sdf->Get<T>(name) : fallback;
  }

  void UpdateJaws()
  {
    if (!driver_ || !left_ || !right_) {
      return;
    }
    const double position = std::clamp(
      offset_ + multiplier_ * driver_->Position(0), lower_limit_, upper_limit_);
    // Do not use Joint::SetPosition here.  It kinematically teleports a
    // collision body every simulation tick.  If a cube is between the jaws,
    // ODE then resolves the resulting deep penetration with an impulse large
    // enough to destabilise the whole arm.  A bounded PD effort is the actual
    // jaw actuator: it closes until contact, holds the cube through friction,
    // and lets contact physics determine the final jaw positions.
    DriveJaw(left_, position);
    DriveJaw(right_, position);
  }

  void DriveJaw(const gazebo::physics::JointPtr & jaw, double target) const
  {
    const double effort = std::clamp(
      position_kp_ * (target - jaw->Position(0)) -
      velocity_kd_ * jaw->GetVelocity(0),
      -max_force_, max_force_);
    jaw->SetForce(0, effort);
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::JointPtr driver_;
  gazebo::physics::JointPtr left_;
  gazebo::physics::JointPtr right_;
  gazebo::event::ConnectionPtr update_connection_;
  double multiplier_{-0.031831};
  double offset_{0.05};
  double lower_limit_{0.0};
  double upper_limit_{0.05};
  double position_kp_{300.0};
  double velocity_kd_{2.0};
  double max_force_{8.0};
};

GZ_REGISTER_MODEL_PLUGIN(GripperMimicPlugin)
}  // namespace gazebo_classic_gripper_mimic

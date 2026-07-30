std_msgs/Header header

string name
string category

geometry_msgs/PoseStamped pose
geometry_msgs/Vector3 size
# Pixel centre in the source image (x=u, y=v); z is unused.
geometry_msgs/Point pixel_center

float32 confidence
uint32 track_id

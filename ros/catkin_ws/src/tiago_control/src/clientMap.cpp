
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/OccupancyGrid.h"
#include "std_msgs/Header.h"
#include "nav_msgs/MapMetaData.h"
#include "nav_msgs/GetMap.h"
#include "geometry_msgs/PoseStamped.h"

nav_msgs::OccupancyGrid map;
ros::Publisher navgoal;

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
  std_msgs::Header header = msg->header;
  nav_msgs::MapMetaData info = msg->info;
  nav_msgs::MapMetaData infoOrig = map.info;
  ROS_INFO("Got      map %d %d", info.width, info.height);
  ROS_INFO("Original map %d %d", infoOrig.width, infoOrig.height);

  geometry_msgs::PoseStamped goal;
  goal.header.seq = 10;
  goal.header.stamp = ros::Time::now();
  goal.header.frame_id = "map";
  goal.pose.position.x = 0.280059704899;
  goal.pose.position.y = -6.23563719551;
  goal.pose.position.y = 0.0;
  goal.pose.orientation.x = 0.0;
  goal.pose.orientation.y = 0.0;
  goal.pose.orientation.z = 0.68885;
  goal.pose.orientation.w = 1.0;
  navgoal.publish(goal);
//Map map(info.width, info.height);
// for (unsigned int x = 0; x < info.width; x++)
//    for (unsigned int y = 0; y < info.height; y++)
//      map.Insert(Cell(x,y,info.width,msg->data[x+ info.width * y]));
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "getterMap");
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<nav_msgs::GetMap>("static_map");
  //ros::ServiceClient client = n.serviceClient<nav_msgs::GetMap>("dynamic_map");
  navgoal = n.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 1);
  nav_msgs::GetMap srv_map;
  if (client.call(srv_map))
  {
    map = srv_map.response.map;
    nav_msgs::MapMetaData info = map.info;
	ROS_INFO("Got original map %d %d", info.width, info.height);
  }
  else
  {
    ROS_ERROR("Failed to call service dynamic_map");
    return 1;
  }
  ros::Subscriber sub = n.subscribe("map", 1, mapCallback);
  ros::spin();
  return 0;
}
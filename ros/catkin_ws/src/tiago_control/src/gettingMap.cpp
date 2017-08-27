#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/OccupancyGrid.h"
#include "std_msgs/Header.h"
#include "nav_msgs/MapMetaData.h"
#include "nav_msgs/GetMap.h"

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
  std_msgs::Header header = msg->header;
  nav_msgs::MapMetaData info = msg->info;
  ROS_INFO("Got map %d %d", info.width, info.height);
//Map map(info.width, info.height);
// for (unsigned int x = 0; x < info.width; x++)
//    for (unsigned int y = 0; y < info.height; y++)
//      map.Insert(Cell(x,y,info.width,msg->data[x+ info.width * y]));
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "mapnodeinfo");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("map", 1, mapCallback);
  ros::spin();

  return 0;
}
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/OccupancyGrid.h"
#include "map_msgs/OccupancyGridUpdate.h"
#include "std_msgs/Header.h"
#include "nav_msgs/MapMetaData.h"
#include "nav_msgs/GetMap.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include <math.h>
#include <unistd.h>

nav_msgs::OccupancyGrid map;
ros::Publisher navgoal;
geometry_msgs::PoseStamped goal;

//the resolution and origin are found in ~/.pal/tiago_maps/config
//map size: 608 672
//resolution: 0.050000 meters/pixel
//origin: [-15.400000, -20.200000, 0.000000] //aka uuper left corner in Rviz
//-> lower right corner = [608*0.05+(-15.4), 672×0.05 + (−20.2)] 
//-> lower right corner = [30.4+(−15.4), 33.6 + (−20.2) ]
//-> lower right corner = [15, 13.4]
double Ax = 0,Bx = 608;
double Ay = 0,By = 672;
double ax = -15.4, bx = 15.0;
double ay = -20.2, by = 13.4;

void chooseNewDestination(){
	for(int j = 0;j<map.info.height;j++)
	for(int i = 0;i<map.info.width;i++)
		{
			if(map.data[i+ map.info.width * j] == 0){
				double xMap = ((double)i -Ax) *(bx-ax)/(Bx-Ax) +ax;
				double yMap = ((double)j -Ay) *(by-ay)/(By-Ay) +ay;
				goal.pose.position.x = xMap;
  				goal.pose.position.y = yMap;
				map.data[i+ map.info.width * j] = 100;
				return;
			}
		}
}

void poseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
  
  double xestim = msg->pose.pose.position.x;
  double yestim = msg->pose.pose.position.y;
  navgoal.publish(goal);
  if (sqrt(pow(goal.pose.position.x - xestim,2.0) +pow(goal.pose.position.y - yestim,2.0)) < 1.2){
	chooseNewDestination();
  }

  for(int x = 0;x<map.info.width;x++)
	for(int y = 0;y<map.info.height;y++)
		{
			if(map.data[x+ map.info.width * y] == 150)
				continue;
			double xMap = ((double)x -Ax) *(bx-ax)/(Bx-Ax) +ax;
			double yMap = ((double)y -Ay) *(by-ay)/(By-Ay) +ay;
			if (sqrt(pow(xMap - xestim,2.0) +pow(yMap - yestim,2.0)) < 0.3){ 
				map.data[x+ map.info.width * y] = 100	; 		
			}
		}
}

void createGoal(){
  goal.header.stamp = ros::Time::now();
  goal.pose.position.x = 0.31059704899;
  goal.pose.position.y = -10.508319551;
  goal.pose.position.z = 0.0;
  goal.pose.orientation.x = 0.0;
  goal.pose.orientation.y = 0.0;
  goal.pose.orientation.z = 0.68885;
  goal.pose.orientation.w = 1.0;
}

void mapCallback2(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
	int width = msg->info.width;
	int height = msg->info.height;
	ROS_INFO("Got map %d %d", width, height);
	if(width != map.info.width || height != map.info.height){
		ROS_INFO("error. nee map size is now allowed");
		exit(-1);
	}
	for (unsigned int x = 0; x < width; x++)
		for (unsigned int y = 0; y < height; y++){
			if(msg->data[x+ width * y] >127){
				map.data[x+ width * y] = msg->data[x+ width * y];
			}
			if(msg->data[x+ width * y] ==0 && map.data[x+ width * y]!=100){	
				map.data[x+ width * y] = msg->data[x+ width * y];
			}
	}
}

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "getterMap");
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<nav_msgs::GetMap>("static_map");
  navgoal = n.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 1);
  createGoal();
  navgoal.publish(goal);
  usleep(1000000);
  nav_msgs::GetMap srv_map;
  if (client.call(srv_map)){
    map = srv_map.response.map;
    nav_msgs::MapMetaData info = map.info;
	ROS_INFO("Got original map %d %d", info.width, info.height);
  }else{
    return 1;
  }
  ros::Subscriber sub = n.subscribe("amcl_pose", 1, poseCallback);
  ros::Subscriber sub2 = n.subscribe("move_base/global_costmap/costmap", 1, mapCallback2);
  ros::spin();
  return 0;
}	

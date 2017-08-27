// Generated by gencpp from file play_motion_msgs/MotionInfo.msg
// DO NOT EDIT!


#ifndef PLAY_MOTION_MSGS_MESSAGE_MOTIONINFO_H
#define PLAY_MOTION_MSGS_MESSAGE_MOTIONINFO_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace play_motion_msgs
{
template <class ContainerAllocator>
struct MotionInfo_
{
  typedef MotionInfo_<ContainerAllocator> Type;

  MotionInfo_()
    : name()
    , joints()
    , duration()  {
    }
  MotionInfo_(const ContainerAllocator& _alloc)
    : name(_alloc)
    , joints(_alloc)
    , duration()  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _name_type;
  _name_type name;

   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > , typename ContainerAllocator::template rebind<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::other >  _joints_type;
  _joints_type joints;

   typedef ros::Duration _duration_type;
  _duration_type duration;




  typedef boost::shared_ptr< ::play_motion_msgs::MotionInfo_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::play_motion_msgs::MotionInfo_<ContainerAllocator> const> ConstPtr;

}; // struct MotionInfo_

typedef ::play_motion_msgs::MotionInfo_<std::allocator<void> > MotionInfo;

typedef boost::shared_ptr< ::play_motion_msgs::MotionInfo > MotionInfoPtr;
typedef boost::shared_ptr< ::play_motion_msgs::MotionInfo const> MotionInfoConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::play_motion_msgs::MotionInfo_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace play_motion_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'play_motion_msgs': ['/home/iftimie/tiago_public_ws/src/play_motion/play_motion_msgs/msg', '/home/iftimie/tiago_public_ws/devel/.private/play_motion_msgs/share/play_motion_msgs/msg'], 'actionlib_msgs': ['/opt/ros/indigo/share/actionlib_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::play_motion_msgs::MotionInfo_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::play_motion_msgs::MotionInfo_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::play_motion_msgs::MotionInfo_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
{
  static const char* value()
  {
    return "12fa5a438744c4ad98a7da64c759d1bd";
  }

  static const char* value(const ::play_motion_msgs::MotionInfo_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x12fa5a438744c4adULL;
  static const uint64_t static_value2 = 0x98a7da64c759d1bdULL;
};

template<class ContainerAllocator>
struct DataType< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
{
  static const char* value()
  {
    return "play_motion_msgs/MotionInfo";
  }

  static const char* value(const ::play_motion_msgs::MotionInfo_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string name\n\
string[] joints\n\
duration duration\n\
";
  }

  static const char* value(const ::play_motion_msgs::MotionInfo_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.name);
      stream.next(m.joints);
      stream.next(m.duration);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct MotionInfo_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::play_motion_msgs::MotionInfo_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::play_motion_msgs::MotionInfo_<ContainerAllocator>& v)
  {
    s << indent << "name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.name);
    s << indent << "joints[]" << std::endl;
    for (size_t i = 0; i < v.joints.size(); ++i)
    {
      s << indent << "  joints[" << i << "]: ";
      Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.joints[i]);
    }
    s << indent << "duration: ";
    Printer<ros::Duration>::stream(s, indent + "  ", v.duration);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PLAY_MOTION_MSGS_MESSAGE_MOTIONINFO_H

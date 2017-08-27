// Generated by gencpp from file pal_common_msgs/JointCurrent.msg
// DO NOT EDIT!


#ifndef PAL_COMMON_MSGS_MESSAGE_JOINTCURRENT_H
#define PAL_COMMON_MSGS_MESSAGE_JOINTCURRENT_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace pal_common_msgs
{
template <class ContainerAllocator>
struct JointCurrent_
{
  typedef JointCurrent_<ContainerAllocator> Type;

  JointCurrent_()
    : joints()
    , current_limit(0.0)  {
    }
  JointCurrent_(const ContainerAllocator& _alloc)
    : joints(_alloc)
    , current_limit(0.0)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _joints_type;
  _joints_type joints;

   typedef float _current_limit_type;
  _current_limit_type current_limit;




  typedef boost::shared_ptr< ::pal_common_msgs::JointCurrent_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pal_common_msgs::JointCurrent_<ContainerAllocator> const> ConstPtr;

}; // struct JointCurrent_

typedef ::pal_common_msgs::JointCurrent_<std::allocator<void> > JointCurrent;

typedef boost::shared_ptr< ::pal_common_msgs::JointCurrent > JointCurrentPtr;
typedef boost::shared_ptr< ::pal_common_msgs::JointCurrent const> JointCurrentConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pal_common_msgs::JointCurrent_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace pal_common_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'actionlib_msgs': ['/opt/ros/indigo/share/actionlib_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'pal_common_msgs': ['/home/iftimie/tiago_public_ws/devel/.private/pal_common_msgs/share/pal_common_msgs/msg', '/home/iftimie/tiago_public_ws/src/pal_msgs/pal_common_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pal_common_msgs::JointCurrent_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_common_msgs::JointCurrent_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_common_msgs::JointCurrent_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
{
  static const char* value()
  {
    return "aa38356f1b4f7b710d0060415affb648";
  }

  static const char* value(const ::pal_common_msgs::JointCurrent_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xaa38356f1b4f7b71ULL;
  static const uint64_t static_value2 = 0x0d0060415affb648ULL;
};

template<class ContainerAllocator>
struct DataType< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pal_common_msgs/JointCurrent";
  }

  static const char* value(const ::pal_common_msgs::JointCurrent_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Joints or group name of joints to activate or deactivate\n\
string joints\n\
\n\
# From 0.0 to 1.0 max current to aply to the joint\n\
float32 current_limit\n\
";
  }

  static const char* value(const ::pal_common_msgs::JointCurrent_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.joints);
      stream.next(m.current_limit);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct JointCurrent_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pal_common_msgs::JointCurrent_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pal_common_msgs::JointCurrent_<ContainerAllocator>& v)
  {
    s << indent << "joints: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.joints);
    s << indent << "current_limit: ";
    Printer<float>::stream(s, indent + "  ", v.current_limit);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PAL_COMMON_MSGS_MESSAGE_JOINTCURRENT_H

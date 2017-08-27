// Generated by gencpp from file pal_detection_msgs/RecognizedActions.msg
// DO NOT EDIT!


#ifndef PAL_DETECTION_MSGS_MESSAGE_RECOGNIZEDACTIONS_H
#define PAL_DETECTION_MSGS_MESSAGE_RECOGNIZEDACTIONS_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace pal_detection_msgs
{
template <class ContainerAllocator>
struct RecognizedActions_
{
  typedef RecognizedActions_<ContainerAllocator> Type;

  RecognizedActions_()
    : header()
    , action_name()
    , u()
    , v()  {
    }
  RecognizedActions_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , action_name(_alloc)
    , u(_alloc)
    , v(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > , typename ContainerAllocator::template rebind<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::other >  _action_name_type;
  _action_name_type action_name;

   typedef std::vector<int64_t, typename ContainerAllocator::template rebind<int64_t>::other >  _u_type;
  _u_type u;

   typedef std::vector<int64_t, typename ContainerAllocator::template rebind<int64_t>::other >  _v_type;
  _v_type v;




  typedef boost::shared_ptr< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> const> ConstPtr;

}; // struct RecognizedActions_

typedef ::pal_detection_msgs::RecognizedActions_<std::allocator<void> > RecognizedActions;

typedef boost::shared_ptr< ::pal_detection_msgs::RecognizedActions > RecognizedActionsPtr;
typedef boost::shared_ptr< ::pal_detection_msgs::RecognizedActions const> RecognizedActionsConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace pal_detection_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/indigo/share/geometry_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/indigo/share/sensor_msgs/cmake/../msg'], 'pal_detection_msgs': ['/home/iftimie/tiago_public_ws/src/pal_msgs/pal_detection_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
{
  static const char* value()
  {
    return "e6c6e85fc615700151c9f88b60277903";
  }

  static const char* value(const ::pal_detection_msgs::RecognizedActions_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xe6c6e85fc6157001ULL;
  static const uint64_t static_value2 = 0x51c9f88b60277903ULL;
};

template<class ContainerAllocator>
struct DataType< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pal_detection_msgs/RecognizedActions";
  }

  static const char* value(const ::pal_detection_msgs::RecognizedActions_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
\n\
string[] action_name    # name of the recognized actions\n\
int64[] u                # (u, v) are the pixel coordinates where\n\
int64[] v                # the actions have been detected\n\
\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
";
  }

  static const char* value(const ::pal_detection_msgs::RecognizedActions_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.action_name);
      stream.next(m.u);
      stream.next(m.v);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct RecognizedActions_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pal_detection_msgs::RecognizedActions_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pal_detection_msgs::RecognizedActions_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "action_name[]" << std::endl;
    for (size_t i = 0; i < v.action_name.size(); ++i)
    {
      s << indent << "  action_name[" << i << "]: ";
      Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.action_name[i]);
    }
    s << indent << "u[]" << std::endl;
    for (size_t i = 0; i < v.u.size(); ++i)
    {
      s << indent << "  u[" << i << "]: ";
      Printer<int64_t>::stream(s, indent + "  ", v.u[i]);
    }
    s << indent << "v[]" << std::endl;
    for (size_t i = 0; i < v.v.size(); ++i)
    {
      s << indent << "  v[" << i << "]: ";
      Printer<int64_t>::stream(s, indent + "  ", v.v[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // PAL_DETECTION_MSGS_MESSAGE_RECOGNIZEDACTIONS_H

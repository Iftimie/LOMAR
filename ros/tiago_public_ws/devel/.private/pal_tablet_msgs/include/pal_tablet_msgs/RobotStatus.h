// Generated by gencpp from file pal_tablet_msgs/RobotStatus.msg
// DO NOT EDIT!


#ifndef PAL_TABLET_MSGS_MESSAGE_ROBOTSTATUS_H
#define PAL_TABLET_MSGS_MESSAGE_ROBOTSTATUS_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <pal_tablet_msgs/FunctionalityStatus.h>

namespace pal_tablet_msgs
{
template <class ContainerAllocator>
struct RobotStatus_
{
  typedef RobotStatus_<ContainerAllocator> Type;

  RobotStatus_()
    : messages()  {
    }
  RobotStatus_(const ContainerAllocator& _alloc)
    : messages(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::pal_tablet_msgs::FunctionalityStatus_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::pal_tablet_msgs::FunctionalityStatus_<ContainerAllocator> >::other >  _messages_type;
  _messages_type messages;




  typedef boost::shared_ptr< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> const> ConstPtr;

}; // struct RobotStatus_

typedef ::pal_tablet_msgs::RobotStatus_<std::allocator<void> > RobotStatus;

typedef boost::shared_ptr< ::pal_tablet_msgs::RobotStatus > RobotStatusPtr;
typedef boost::shared_ptr< ::pal_tablet_msgs::RobotStatus const> RobotStatusConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace pal_tablet_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'pal_tablet_msgs': ['/home/iftimie/tiago_public_ws/src/pal_msgs/pal_tablet_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
{
  static const char* value()
  {
    return "3495bf9d81b79deaa82f3652871818c5";
  }

  static const char* value(const ::pal_tablet_msgs::RobotStatus_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x3495bf9d81b79deaULL;
  static const uint64_t static_value2 = 0xa82f3652871818c5ULL;
};

template<class ContainerAllocator>
struct DataType< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pal_tablet_msgs/RobotStatus";
  }

  static const char* value(const ::pal_tablet_msgs::RobotStatus_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Robot status msgs for tablet\n\
\n\
pal_tablet_msgs/FunctionalityStatus[] messages \n\
\n\
================================================================================\n\
MSG: pal_tablet_msgs/FunctionalityStatus\n\
# Functionality status message\n\
\n\
std_msgs/String   errMessage\n\
std_msgs/Bool     errStatus\n\
\n\
================================================================================\n\
MSG: std_msgs/String\n\
string data\n\
\n\
================================================================================\n\
MSG: std_msgs/Bool\n\
bool data\n\
";
  }

  static const char* value(const ::pal_tablet_msgs::RobotStatus_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.messages);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct RobotStatus_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pal_tablet_msgs::RobotStatus_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pal_tablet_msgs::RobotStatus_<ContainerAllocator>& v)
  {
    s << indent << "messages[]" << std::endl;
    for (size_t i = 0; i < v.messages.size(); ++i)
    {
      s << indent << "  messages[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::pal_tablet_msgs::FunctionalityStatus_<ContainerAllocator> >::stream(s, indent + "    ", v.messages[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // PAL_TABLET_MSGS_MESSAGE_ROBOTSTATUS_H

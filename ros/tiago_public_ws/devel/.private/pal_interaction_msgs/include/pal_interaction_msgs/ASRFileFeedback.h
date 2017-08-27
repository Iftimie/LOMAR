// Generated by gencpp from file pal_interaction_msgs/ASRFileFeedback.msg
// DO NOT EDIT!


#ifndef PAL_INTERACTION_MSGS_MESSAGE_ASRFILEFEEDBACK_H
#define PAL_INTERACTION_MSGS_MESSAGE_ASRFILEFEEDBACK_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <pal_interaction_msgs/asrresult.h>

namespace pal_interaction_msgs
{
template <class ContainerAllocator>
struct ASRFileFeedback_
{
  typedef ASRFileFeedback_<ContainerAllocator> Type;

  ASRFileFeedback_()
    : recognised_utterance()  {
    }
  ASRFileFeedback_(const ContainerAllocator& _alloc)
    : recognised_utterance(_alloc)  {
  (void)_alloc;
    }



   typedef  ::pal_interaction_msgs::asrresult_<ContainerAllocator>  _recognised_utterance_type;
  _recognised_utterance_type recognised_utterance;




  typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> const> ConstPtr;

}; // struct ASRFileFeedback_

typedef ::pal_interaction_msgs::ASRFileFeedback_<std::allocator<void> > ASRFileFeedback;

typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileFeedback > ASRFileFeedbackPtr;
typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileFeedback const> ASRFileFeedbackConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace pal_interaction_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'actionlib_msgs': ['/opt/ros/indigo/share/actionlib_msgs/cmake/../msg'], 'pal_interaction_msgs': ['/home/iftimie/tiago_public_ws/src/pal_msgs/pal_interaction_msgs/msg', '/home/iftimie/tiago_public_ws/devel/.private/pal_interaction_msgs/share/pal_interaction_msgs/msg'], 'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "e8f3da6b7eb47ddaa66e1eca614ca0be";
  }

  static const char* value(const ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xe8f3da6b7eb47ddaULL;
  static const uint64_t static_value2 = 0xa66e1eca614ca0beULL;
};

template<class ContainerAllocator>
struct DataType< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pal_interaction_msgs/ASRFileFeedback";
  }

  static const char* value(const ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
#feedback definition\n\
# At the inmediate time an utterance is recognised\n\
# it is published here.\n\
asrresult recognised_utterance\n\
\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/asrresult\n\
## Message that containes the recognized utterance.\n\
## Confidence values\n\
int8 CONFIDENCE_UNDEFINED = -1\n\
int8 CONFIDENCE_POOR = 1\n\
int8 CONFIDENCE_LOW  = 2\n\
int8 CONFIDENCE_GOOD = 3\n\
int8 CONFIDENCE_MAX  = 4\n\
\n\
# ASR result messages used by RosRecognizerServer\n\
\n\
# text recognized\n\
string text\n\
\n\
# confidence with values from POOR to MAX\n\
int8 confidence\n\
\n\
# start and end of the recognizer uterance.\n\
time start\n\
time end\n\
\n\
# list of recognized tags\n\
# key value pairs of strings extracted from the text\n\
# given the action tags placed in the grammar.\n\
actiontag[] tags\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/actiontag\n\
# Action tag contaings the key/value information genertated by parsing the recognised text with a JSGF grammar \n\
\n\
string key\n\
string value\n\
";
  }

  static const char* value(const ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.recognised_utterance);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ASRFileFeedback_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pal_interaction_msgs::ASRFileFeedback_<ContainerAllocator>& v)
  {
    s << indent << "recognised_utterance: ";
    s << std::endl;
    Printer< ::pal_interaction_msgs::asrresult_<ContainerAllocator> >::stream(s, indent + "  ", v.recognised_utterance);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PAL_INTERACTION_MSGS_MESSAGE_ASRFILEFEEDBACK_H

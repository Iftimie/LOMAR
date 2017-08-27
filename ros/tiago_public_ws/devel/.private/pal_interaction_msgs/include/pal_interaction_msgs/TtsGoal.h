// Generated by gencpp from file pal_interaction_msgs/TtsGoal.msg
// DO NOT EDIT!


#ifndef PAL_INTERACTION_MSGS_MESSAGE_TTSGOAL_H
#define PAL_INTERACTION_MSGS_MESSAGE_TTSGOAL_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <pal_interaction_msgs/I18nText.h>
#include <pal_interaction_msgs/TtsText.h>

namespace pal_interaction_msgs
{
template <class ContainerAllocator>
struct TtsGoal_
{
  typedef TtsGoal_<ContainerAllocator> Type;

  TtsGoal_()
    : text()
    , rawtext()
    , speakerName()
    , wait_before_speaking(0.0)  {
    }
  TtsGoal_(const ContainerAllocator& _alloc)
    : text(_alloc)
    , rawtext(_alloc)
    , speakerName(_alloc)
    , wait_before_speaking(0.0)  {
  (void)_alloc;
    }



   typedef  ::pal_interaction_msgs::I18nText_<ContainerAllocator>  _text_type;
  _text_type text;

   typedef  ::pal_interaction_msgs::TtsText_<ContainerAllocator>  _rawtext_type;
  _rawtext_type rawtext;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _speakerName_type;
  _speakerName_type speakerName;

   typedef double _wait_before_speaking_type;
  _wait_before_speaking_type wait_before_speaking;




  typedef boost::shared_ptr< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> const> ConstPtr;

}; // struct TtsGoal_

typedef ::pal_interaction_msgs::TtsGoal_<std::allocator<void> > TtsGoal;

typedef boost::shared_ptr< ::pal_interaction_msgs::TtsGoal > TtsGoalPtr;
typedef boost::shared_ptr< ::pal_interaction_msgs::TtsGoal const> TtsGoalConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
{
  static const char* value()
  {
    return "9c88bf4a4d119474b8ae97c98892bc78";
  }

  static const char* value(const ::pal_interaction_msgs::TtsGoal_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x9c88bf4a4d119474ULL;
  static const uint64_t static_value2 = 0xb8ae97c98892bc78ULL;
};

template<class ContainerAllocator>
struct DataType< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pal_interaction_msgs/TtsGoal";
  }

  static const char* value(const ::pal_interaction_msgs::TtsGoal_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
## goal definition\n\
\n\
# utterance will contain indications to construct\n\
# the text to be spoken.\n\
# It must be specified in a section/key format \n\
# for internationalisation. The actual text will\n\
# be obtained from configuration files as in pal_tts_cfg pkg.\n\
 \n\
I18nText text\n\
TtsText rawtext\n\
\n\
# This is to suggest a voice name to the \n\
# tts engine. For the same language we might have\n\
# a variety of voices available, and this variable \n\
# is to choose one among them. \n\
# (not implemented yet)\n\
string speakerName\n\
\n\
# Time to wait before synthesising the text itself.\n\
# It can be used to produced delayed speech.\n\
float64 wait_before_speaking\n\
\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/I18nText\n\
# section/key is used as in examples in the pal_tts_cfg pkg.\n\
string section\n\
string key\n\
\n\
# language id, must be speficied using RFC 3066\n\
string lang_id\n\
\n\
# arguments contain the values by which \n\
# occurrences of '%s' will be replaced in the \n\
# main text.\n\
# This only supports up to 2 arguments and no recursion.\n\
# However, recursion and more argumnents are\n\
# planned to be supported in the future.\n\
I18nArgument[] arguments\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/I18nArgument\n\
# section key, override the value in expanded.\n\
# Use expanded for text that do not need expansion.\n\
# Please note that expanded here will not be translated \n\
# to any language.\n\
\n\
string section\n\
string key\n\
string expanded\n\
\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/TtsText\n\
# this message is to specify \n\
# raw text to the TTS server. \n\
\n\
string text\n\
\n\
# Language id in RFC 3066 format\n\
# This field is mandatory\n\
string lang_id\n\
";
  }

  static const char* value(const ::pal_interaction_msgs::TtsGoal_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.text);
      stream.next(m.rawtext);
      stream.next(m.speakerName);
      stream.next(m.wait_before_speaking);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct TtsGoal_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pal_interaction_msgs::TtsGoal_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pal_interaction_msgs::TtsGoal_<ContainerAllocator>& v)
  {
    s << indent << "text: ";
    s << std::endl;
    Printer< ::pal_interaction_msgs::I18nText_<ContainerAllocator> >::stream(s, indent + "  ", v.text);
    s << indent << "rawtext: ";
    s << std::endl;
    Printer< ::pal_interaction_msgs::TtsText_<ContainerAllocator> >::stream(s, indent + "  ", v.rawtext);
    s << indent << "speakerName: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.speakerName);
    s << indent << "wait_before_speaking: ";
    Printer<double>::stream(s, indent + "  ", v.wait_before_speaking);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PAL_INTERACTION_MSGS_MESSAGE_TTSGOAL_H

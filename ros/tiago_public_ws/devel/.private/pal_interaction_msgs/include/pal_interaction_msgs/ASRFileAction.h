// Generated by gencpp from file pal_interaction_msgs/ASRFileAction.msg
// DO NOT EDIT!


#ifndef PAL_INTERACTION_MSGS_MESSAGE_ASRFILEACTION_H
#define PAL_INTERACTION_MSGS_MESSAGE_ASRFILEACTION_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <pal_interaction_msgs/ASRFileActionGoal.h>
#include <pal_interaction_msgs/ASRFileActionResult.h>
#include <pal_interaction_msgs/ASRFileActionFeedback.h>

namespace pal_interaction_msgs
{
template <class ContainerAllocator>
struct ASRFileAction_
{
  typedef ASRFileAction_<ContainerAllocator> Type;

  ASRFileAction_()
    : action_goal()
    , action_result()
    , action_feedback()  {
    }
  ASRFileAction_(const ContainerAllocator& _alloc)
    : action_goal(_alloc)
    , action_result(_alloc)
    , action_feedback(_alloc)  {
  (void)_alloc;
    }



   typedef  ::pal_interaction_msgs::ASRFileActionGoal_<ContainerAllocator>  _action_goal_type;
  _action_goal_type action_goal;

   typedef  ::pal_interaction_msgs::ASRFileActionResult_<ContainerAllocator>  _action_result_type;
  _action_result_type action_result;

   typedef  ::pal_interaction_msgs::ASRFileActionFeedback_<ContainerAllocator>  _action_feedback_type;
  _action_feedback_type action_feedback;




  typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> const> ConstPtr;

}; // struct ASRFileAction_

typedef ::pal_interaction_msgs::ASRFileAction_<std::allocator<void> > ASRFileAction;

typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileAction > ASRFileActionPtr;
typedef boost::shared_ptr< ::pal_interaction_msgs::ASRFileAction const> ASRFileActionConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
{
  static const char* value()
  {
    return "755ced780decb7cb4a33fd3b127d2d2e";
  }

  static const char* value(const ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x755ced780decb7cbULL;
  static const uint64_t static_value2 = 0x4a33fd3b127d2d2eULL;
};

template<class ContainerAllocator>
struct DataType< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pal_interaction_msgs/ASRFileAction";
  }

  static const char* value(const ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
\n\
ASRFileActionGoal action_goal\n\
ASRFileActionResult action_result\n\
ASRFileActionFeedback action_feedback\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/ASRFileActionGoal\n\
# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
\n\
Header header\n\
actionlib_msgs/GoalID goal_id\n\
ASRFileGoal goal\n\
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
\n\
================================================================================\n\
MSG: actionlib_msgs/GoalID\n\
# The stamp should store the time at which this goal was requested.\n\
# It is used by an action server when it tries to preempt all\n\
# goals that were requested before a certain time\n\
time stamp\n\
\n\
# The id provides a way to associate feedback and\n\
# result message with specific goal requests. The id\n\
# specified must be unique.\n\
string id\n\
\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/ASRFileGoal\n\
# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
##goal definition\n\
# absolute path to the file to be recognised.\n\
# format has to be PCM 16 bits signed integer\n\
string file\n\
# language id. (i.e., en_US, es_ES, ...)\n\
string lang_id\n\
# grammar name\n\
string grammar\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/ASRFileActionResult\n\
# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
\n\
Header header\n\
actionlib_msgs/GoalStatus status\n\
ASRFileResult result\n\
\n\
================================================================================\n\
MSG: actionlib_msgs/GoalStatus\n\
GoalID goal_id\n\
uint8 status\n\
uint8 PENDING         = 0   # The goal has yet to be processed by the action server\n\
uint8 ACTIVE          = 1   # The goal is currently being processed by the action server\n\
uint8 PREEMPTED       = 2   # The goal received a cancel request after it started executing\n\
                            #   and has since completed its execution (Terminal State)\n\
uint8 SUCCEEDED       = 3   # The goal was achieved successfully by the action server (Terminal State)\n\
uint8 ABORTED         = 4   # The goal was aborted during execution by the action server due\n\
                            #    to some failure (Terminal State)\n\
uint8 REJECTED        = 5   # The goal was rejected by the action server without being processed,\n\
                            #    because the goal was unattainable or invalid (Terminal State)\n\
uint8 PREEMPTING      = 6   # The goal received a cancel request after it started executing\n\
                            #    and has not yet completed execution\n\
uint8 RECALLING       = 7   # The goal received a cancel request before it started executing,\n\
                            #    but the action server has not yet confirmed that the goal is canceled\n\
uint8 RECALLED        = 8   # The goal received a cancel request before it started executing\n\
                            #    and was successfully cancelled (Terminal State)\n\
uint8 LOST            = 9   # An action client can determine that a goal is LOST. This should not be\n\
                            #    sent over the wire by an action server\n\
\n\
#Allow for the user to associate a string with GoalStatus for debugging\n\
string text\n\
\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/ASRFileResult\n\
# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
##result definition\n\
# same path as specified in goal variable file\n\
string file\n\
# error/warning messages \n\
string msg\n\
# vector of results\n\
asrresult[] recognised_utterances\n\
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
================================================================================\n\
MSG: pal_interaction_msgs/ASRFileActionFeedback\n\
# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
\n\
Header header\n\
actionlib_msgs/GoalStatus status\n\
ASRFileFeedback feedback\n\
\n\
================================================================================\n\
MSG: pal_interaction_msgs/ASRFileFeedback\n\
# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
#feedback definition\n\
# At the inmediate time an utterance is recognised\n\
# it is published here.\n\
asrresult recognised_utterance\n\
\n\
";
  }

  static const char* value(const ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.action_goal);
      stream.next(m.action_result);
      stream.next(m.action_feedback);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ASRFileAction_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pal_interaction_msgs::ASRFileAction_<ContainerAllocator>& v)
  {
    s << indent << "action_goal: ";
    s << std::endl;
    Printer< ::pal_interaction_msgs::ASRFileActionGoal_<ContainerAllocator> >::stream(s, indent + "  ", v.action_goal);
    s << indent << "action_result: ";
    s << std::endl;
    Printer< ::pal_interaction_msgs::ASRFileActionResult_<ContainerAllocator> >::stream(s, indent + "  ", v.action_result);
    s << indent << "action_feedback: ";
    s << std::endl;
    Printer< ::pal_interaction_msgs::ASRFileActionFeedback_<ContainerAllocator> >::stream(s, indent + "  ", v.action_feedback);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PAL_INTERACTION_MSGS_MESSAGE_ASRFILEACTION_H

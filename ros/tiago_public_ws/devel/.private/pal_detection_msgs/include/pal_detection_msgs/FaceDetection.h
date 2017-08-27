// Generated by gencpp from file pal_detection_msgs/FaceDetection.msg
// DO NOT EDIT!


#ifndef PAL_DETECTION_MSGS_MESSAGE_FACEDETECTION_H
#define PAL_DETECTION_MSGS_MESSAGE_FACEDETECTION_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Point32.h>

namespace pal_detection_msgs
{
template <class ContainerAllocator>
struct FaceDetection_
{
  typedef FaceDetection_<ContainerAllocator> Type;

  FaceDetection_()
    : x(0)
    , y(0)
    , width(0)
    , height(0)
    , eyesLocated(false)
    , leftEyeX(0)
    , leftEyeY(0)
    , rightEyeX(0)
    , rightEyeY(0)
    , position()
    , name()
    , confidence(0.0)
    , expression()
    , expression_confidence(0.0)
    , emotion_anger_confidence(0.0)
    , emotion_disgust_confidence(0.0)
    , emotion_fear_confidence(0.0)
    , emotion_happiness_confidence(0.0)
    , emotion_neutral_confidence(0.0)
    , emotion_sadness_confidence(0.0)
    , emotion_surprise_confidence(0.0)  {
    }
  FaceDetection_(const ContainerAllocator& _alloc)
    : x(0)
    , y(0)
    , width(0)
    , height(0)
    , eyesLocated(false)
    , leftEyeX(0)
    , leftEyeY(0)
    , rightEyeX(0)
    , rightEyeY(0)
    , position(_alloc)
    , name(_alloc)
    , confidence(0.0)
    , expression(_alloc)
    , expression_confidence(0.0)
    , emotion_anger_confidence(0.0)
    , emotion_disgust_confidence(0.0)
    , emotion_fear_confidence(0.0)
    , emotion_happiness_confidence(0.0)
    , emotion_neutral_confidence(0.0)
    , emotion_sadness_confidence(0.0)
    , emotion_surprise_confidence(0.0)  {
  (void)_alloc;
    }



   typedef int32_t _x_type;
  _x_type x;

   typedef int32_t _y_type;
  _y_type y;

   typedef int32_t _width_type;
  _width_type width;

   typedef int32_t _height_type;
  _height_type height;

   typedef uint8_t _eyesLocated_type;
  _eyesLocated_type eyesLocated;

   typedef int32_t _leftEyeX_type;
  _leftEyeX_type leftEyeX;

   typedef int32_t _leftEyeY_type;
  _leftEyeY_type leftEyeY;

   typedef int32_t _rightEyeX_type;
  _rightEyeX_type rightEyeX;

   typedef int32_t _rightEyeY_type;
  _rightEyeY_type rightEyeY;

   typedef  ::geometry_msgs::Point32_<ContainerAllocator>  _position_type;
  _position_type position;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _name_type;
  _name_type name;

   typedef float _confidence_type;
  _confidence_type confidence;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _expression_type;
  _expression_type expression;

   typedef float _expression_confidence_type;
  _expression_confidence_type expression_confidence;

   typedef float _emotion_anger_confidence_type;
  _emotion_anger_confidence_type emotion_anger_confidence;

   typedef float _emotion_disgust_confidence_type;
  _emotion_disgust_confidence_type emotion_disgust_confidence;

   typedef float _emotion_fear_confidence_type;
  _emotion_fear_confidence_type emotion_fear_confidence;

   typedef float _emotion_happiness_confidence_type;
  _emotion_happiness_confidence_type emotion_happiness_confidence;

   typedef float _emotion_neutral_confidence_type;
  _emotion_neutral_confidence_type emotion_neutral_confidence;

   typedef float _emotion_sadness_confidence_type;
  _emotion_sadness_confidence_type emotion_sadness_confidence;

   typedef float _emotion_surprise_confidence_type;
  _emotion_surprise_confidence_type emotion_surprise_confidence;


    static const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  EXPRESSION_NEUTRAL;
     static const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  EXPRESSION_SMILE;
     static const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  EXPRESSION_RAISED_BROWS;
     static const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  EXPRESSION_EYES_AWAY;
     static const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  EXPRESSION_SQUINTING;
     static const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  EXPRESSION_FROWNING;
 

  typedef boost::shared_ptr< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> const> ConstPtr;

}; // struct FaceDetection_

typedef ::pal_detection_msgs::FaceDetection_<std::allocator<void> > FaceDetection;

typedef boost::shared_ptr< ::pal_detection_msgs::FaceDetection > FaceDetectionPtr;
typedef boost::shared_ptr< ::pal_detection_msgs::FaceDetection const> FaceDetectionConstPtr;

// constants requiring out of line definition

   
   template<typename ContainerAllocator> const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > 
      FaceDetection_<ContainerAllocator>::EXPRESSION_NEUTRAL =
        
          "\"neutral\""
        
        ;
   

   
   template<typename ContainerAllocator> const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > 
      FaceDetection_<ContainerAllocator>::EXPRESSION_SMILE =
        
          "\"smile\""
        
        ;
   

   
   template<typename ContainerAllocator> const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > 
      FaceDetection_<ContainerAllocator>::EXPRESSION_RAISED_BROWS =
        
          "\"raised brows\""
        
        ;
   

   
   template<typename ContainerAllocator> const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > 
      FaceDetection_<ContainerAllocator>::EXPRESSION_EYES_AWAY =
        
          "\"eyes away\""
        
        ;
   

   
   template<typename ContainerAllocator> const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > 
      FaceDetection_<ContainerAllocator>::EXPRESSION_SQUINTING =
        
          "\"squinting\""
        
        ;
   

   
   template<typename ContainerAllocator> const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > 
      FaceDetection_<ContainerAllocator>::EXPRESSION_FROWNING =
        
          "\"frowning\""
        
        ;
   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pal_detection_msgs::FaceDetection_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace pal_detection_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/indigo/share/geometry_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/indigo/share/sensor_msgs/cmake/../msg'], 'pal_detection_msgs': ['/home/iftimie/tiago_public_ws/src/pal_msgs/pal_detection_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "a17fdb6b06e8f6a7b02b1e50262a787c";
  }

  static const char* value(const ::pal_detection_msgs::FaceDetection_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xa17fdb6b06e8f6a7ULL;
  static const uint64_t static_value2 = 0xb02b1e50262a787cULL;
};

template<class ContainerAllocator>
struct DataType< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pal_detection_msgs/FaceDetection";
  }

  static const char* value(const ::pal_detection_msgs::FaceDetection_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
##########################################\n\
#\n\
# Face detection data\n\
#\n\
##########################################\n\
\n\
#####################\n\
# Face bounding box\n\
#####################\n\
# coordinates of the top left corner of the box\n\
int32 x\n\
int32 y\n\
\n\
# width of the box\n\
int32 width\n\
\n\
# height of the box\n\
int32 height\n\
\n\
############################\n\
# Eyes position (if found)\n\
############################\n\
\n\
bool eyesLocated\n\
\n\
int32 leftEyeX\n\
int32 leftEyeY\n\
int32 rightEyeX\n\
int32 rightEyeY\n\
\n\
#############################\n\
# Centre of eyes 3D estimate\n\
#############################\n\
geometry_msgs/Point32 position\n\
\n\
\n\
############################\n\
# Person recognition\n\
############################\n\
\n\
string name\n\
float32 confidence\n\
\n\
############################\n\
# Facial expression\n\
############################\n\
string EXPRESSION_NEUTRAL=\"neutral\"\n\
string EXPRESSION_SMILE=\"smile\"\n\
string EXPRESSION_RAISED_BROWS=\"raised brows\"\n\
string EXPRESSION_EYES_AWAY=\"eyes away\"\n\
string EXPRESSION_SQUINTING=\"squinting\"\n\
string EXPRESSION_FROWNING=\"frowning\"\n\
string expression\n\
float32 expression_confidence\n\
\n\
############################\n\
# Facial emotion\n\
############################\n\
float32 emotion_anger_confidence\n\
float32 emotion_disgust_confidence\n\
float32 emotion_fear_confidence\n\
float32 emotion_happiness_confidence\n\
float32 emotion_neutral_confidence\n\
float32 emotion_sadness_confidence\n\
float32 emotion_surprise_confidence\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Point32\n\
# This contains the position of a point in free space(with 32 bits of precision).\n\
# It is recommeded to use Point wherever possible instead of Point32.  \n\
# \n\
# This recommendation is to promote interoperability.  \n\
#\n\
# This message is designed to take up less space when sending\n\
# lots of points at once, as in the case of a PointCloud.  \n\
\n\
float32 x\n\
float32 y\n\
float32 z\n\
";
  }

  static const char* value(const ::pal_detection_msgs::FaceDetection_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.x);
      stream.next(m.y);
      stream.next(m.width);
      stream.next(m.height);
      stream.next(m.eyesLocated);
      stream.next(m.leftEyeX);
      stream.next(m.leftEyeY);
      stream.next(m.rightEyeX);
      stream.next(m.rightEyeY);
      stream.next(m.position);
      stream.next(m.name);
      stream.next(m.confidence);
      stream.next(m.expression);
      stream.next(m.expression_confidence);
      stream.next(m.emotion_anger_confidence);
      stream.next(m.emotion_disgust_confidence);
      stream.next(m.emotion_fear_confidence);
      stream.next(m.emotion_happiness_confidence);
      stream.next(m.emotion_neutral_confidence);
      stream.next(m.emotion_sadness_confidence);
      stream.next(m.emotion_surprise_confidence);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct FaceDetection_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pal_detection_msgs::FaceDetection_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pal_detection_msgs::FaceDetection_<ContainerAllocator>& v)
  {
    s << indent << "x: ";
    Printer<int32_t>::stream(s, indent + "  ", v.x);
    s << indent << "y: ";
    Printer<int32_t>::stream(s, indent + "  ", v.y);
    s << indent << "width: ";
    Printer<int32_t>::stream(s, indent + "  ", v.width);
    s << indent << "height: ";
    Printer<int32_t>::stream(s, indent + "  ", v.height);
    s << indent << "eyesLocated: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.eyesLocated);
    s << indent << "leftEyeX: ";
    Printer<int32_t>::stream(s, indent + "  ", v.leftEyeX);
    s << indent << "leftEyeY: ";
    Printer<int32_t>::stream(s, indent + "  ", v.leftEyeY);
    s << indent << "rightEyeX: ";
    Printer<int32_t>::stream(s, indent + "  ", v.rightEyeX);
    s << indent << "rightEyeY: ";
    Printer<int32_t>::stream(s, indent + "  ", v.rightEyeY);
    s << indent << "position: ";
    s << std::endl;
    Printer< ::geometry_msgs::Point32_<ContainerAllocator> >::stream(s, indent + "  ", v.position);
    s << indent << "name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.name);
    s << indent << "confidence: ";
    Printer<float>::stream(s, indent + "  ", v.confidence);
    s << indent << "expression: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.expression);
    s << indent << "expression_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.expression_confidence);
    s << indent << "emotion_anger_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.emotion_anger_confidence);
    s << indent << "emotion_disgust_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.emotion_disgust_confidence);
    s << indent << "emotion_fear_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.emotion_fear_confidence);
    s << indent << "emotion_happiness_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.emotion_happiness_confidence);
    s << indent << "emotion_neutral_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.emotion_neutral_confidence);
    s << indent << "emotion_sadness_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.emotion_sadness_confidence);
    s << indent << "emotion_surprise_confidence: ";
    Printer<float>::stream(s, indent + "  ", v.emotion_surprise_confidence);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PAL_DETECTION_MSGS_MESSAGE_FACEDETECTION_H

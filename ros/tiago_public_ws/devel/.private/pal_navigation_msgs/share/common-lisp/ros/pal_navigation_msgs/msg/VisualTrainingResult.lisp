; Auto-generated. Do not edit!


(cl:in-package pal_navigation_msgs-msg)


;//! \htmlinclude VisualTrainingResult.msg.html

(cl:defclass <VisualTrainingResult> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass VisualTrainingResult (<VisualTrainingResult>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <VisualTrainingResult>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'VisualTrainingResult)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name pal_navigation_msgs-msg:<VisualTrainingResult> is deprecated: use pal_navigation_msgs-msg:VisualTrainingResult instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <VisualTrainingResult>) ostream)
  "Serializes a message object of type '<VisualTrainingResult>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <VisualTrainingResult>) istream)
  "Deserializes a message object of type '<VisualTrainingResult>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<VisualTrainingResult>)))
  "Returns string type for a message object of type '<VisualTrainingResult>"
  "pal_navigation_msgs/VisualTrainingResult")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'VisualTrainingResult)))
  "Returns string type for a message object of type 'VisualTrainingResult"
  "pal_navigation_msgs/VisualTrainingResult")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<VisualTrainingResult>)))
  "Returns md5sum for a message object of type '<VisualTrainingResult>"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'VisualTrainingResult)))
  "Returns md5sum for a message object of type 'VisualTrainingResult"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<VisualTrainingResult>)))
  "Returns full string definition for message of type '<VisualTrainingResult>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%#result definition~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'VisualTrainingResult)))
  "Returns full string definition for message of type 'VisualTrainingResult"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%#result definition~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <VisualTrainingResult>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <VisualTrainingResult>))
  "Converts a ROS message object to a list"
  (cl:list 'VisualTrainingResult
))

; Auto-generated. Do not edit!


(cl:in-package pal_common_msgs-msg)


;//! \htmlinclude EmptyResult.msg.html

(cl:defclass <EmptyResult> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass EmptyResult (<EmptyResult>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EmptyResult>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EmptyResult)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name pal_common_msgs-msg:<EmptyResult> is deprecated: use pal_common_msgs-msg:EmptyResult instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EmptyResult>) ostream)
  "Serializes a message object of type '<EmptyResult>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EmptyResult>) istream)
  "Deserializes a message object of type '<EmptyResult>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EmptyResult>)))
  "Returns string type for a message object of type '<EmptyResult>"
  "pal_common_msgs/EmptyResult")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EmptyResult)))
  "Returns string type for a message object of type 'EmptyResult"
  "pal_common_msgs/EmptyResult")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EmptyResult>)))
  "Returns md5sum for a message object of type '<EmptyResult>"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EmptyResult)))
  "Returns md5sum for a message object of type 'EmptyResult"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EmptyResult>)))
  "Returns full string definition for message of type '<EmptyResult>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EmptyResult)))
  "Returns full string definition for message of type 'EmptyResult"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EmptyResult>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EmptyResult>))
  "Converts a ROS message object to a list"
  (cl:list 'EmptyResult
))

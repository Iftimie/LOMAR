; Auto-generated. Do not edit!


(cl:in-package play_motion_msgs-srv)


;//! \htmlinclude ListMotions-request.msg.html

(cl:defclass <ListMotions-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass ListMotions-request (<ListMotions-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ListMotions-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ListMotions-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name play_motion_msgs-srv:<ListMotions-request> is deprecated: use play_motion_msgs-srv:ListMotions-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ListMotions-request>) ostream)
  "Serializes a message object of type '<ListMotions-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ListMotions-request>) istream)
  "Deserializes a message object of type '<ListMotions-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ListMotions-request>)))
  "Returns string type for a service object of type '<ListMotions-request>"
  "play_motion_msgs/ListMotionsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ListMotions-request)))
  "Returns string type for a service object of type 'ListMotions-request"
  "play_motion_msgs/ListMotionsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ListMotions-request>)))
  "Returns md5sum for a message object of type '<ListMotions-request>"
  "151d88e888f918ab21ebda2cef4dd23c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ListMotions-request)))
  "Returns md5sum for a message object of type 'ListMotions-request"
  "151d88e888f918ab21ebda2cef4dd23c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ListMotions-request>)))
  "Returns full string definition for message of type '<ListMotions-request>"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ListMotions-request)))
  "Returns full string definition for message of type 'ListMotions-request"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ListMotions-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ListMotions-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ListMotions-request
))
;//! \htmlinclude ListMotions-response.msg.html

(cl:defclass <ListMotions-response> (roslisp-msg-protocol:ros-message)
  ((motions
    :reader motions
    :initarg :motions
    :type (cl:vector play_motion_msgs-msg:MotionInfo)
   :initform (cl:make-array 0 :element-type 'play_motion_msgs-msg:MotionInfo :initial-element (cl:make-instance 'play_motion_msgs-msg:MotionInfo))))
)

(cl:defclass ListMotions-response (<ListMotions-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ListMotions-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ListMotions-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name play_motion_msgs-srv:<ListMotions-response> is deprecated: use play_motion_msgs-srv:ListMotions-response instead.")))

(cl:ensure-generic-function 'motions-val :lambda-list '(m))
(cl:defmethod motions-val ((m <ListMotions-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-srv:motions-val is deprecated.  Use play_motion_msgs-srv:motions instead.")
  (motions m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ListMotions-response>) ostream)
  "Serializes a message object of type '<ListMotions-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'motions))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'motions))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ListMotions-response>) istream)
  "Deserializes a message object of type '<ListMotions-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'motions) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'motions)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'play_motion_msgs-msg:MotionInfo))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ListMotions-response>)))
  "Returns string type for a service object of type '<ListMotions-response>"
  "play_motion_msgs/ListMotionsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ListMotions-response)))
  "Returns string type for a service object of type 'ListMotions-response"
  "play_motion_msgs/ListMotionsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ListMotions-response>)))
  "Returns md5sum for a message object of type '<ListMotions-response>"
  "151d88e888f918ab21ebda2cef4dd23c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ListMotions-response)))
  "Returns md5sum for a message object of type 'ListMotions-response"
  "151d88e888f918ab21ebda2cef4dd23c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ListMotions-response>)))
  "Returns full string definition for message of type '<ListMotions-response>"
  (cl:format cl:nil "MotionInfo[] motions~%~%================================================================================~%MSG: play_motion_msgs/MotionInfo~%string name~%string[] joints~%duration duration~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ListMotions-response)))
  "Returns full string definition for message of type 'ListMotions-response"
  (cl:format cl:nil "MotionInfo[] motions~%~%================================================================================~%MSG: play_motion_msgs/MotionInfo~%string name~%string[] joints~%duration duration~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ListMotions-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'motions) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ListMotions-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ListMotions-response
    (cl:cons ':motions (motions msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ListMotions)))
  'ListMotions-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ListMotions)))
  'ListMotions-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ListMotions)))
  "Returns string type for a service object of type '<ListMotions>"
  "play_motion_msgs/ListMotions")
; Auto-generated. Do not edit!


(cl:in-package tiago_opencv_tutorial-msg)


;//! \htmlinclude valueMatrix.msg.html

(cl:defclass <valueMatrix> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (value
    :reader value
    :initarg :value
    :type cl:float
    :initform 0.0)
   (tick
    :reader tick
    :initarg :tick
    :type cl:boolean
    :initform cl:nil)
   (option
    :reader option
    :initarg :option
    :type cl:string
    :initform ""))
)

(cl:defclass valueMatrix (<valueMatrix>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <valueMatrix>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'valueMatrix)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tiago_opencv_tutorial-msg:<valueMatrix> is deprecated: use tiago_opencv_tutorial-msg:valueMatrix instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <valueMatrix>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tiago_opencv_tutorial-msg:header-val is deprecated.  Use tiago_opencv_tutorial-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <valueMatrix>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tiago_opencv_tutorial-msg:value-val is deprecated.  Use tiago_opencv_tutorial-msg:value instead.")
  (value m))

(cl:ensure-generic-function 'tick-val :lambda-list '(m))
(cl:defmethod tick-val ((m <valueMatrix>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tiago_opencv_tutorial-msg:tick-val is deprecated.  Use tiago_opencv_tutorial-msg:tick instead.")
  (tick m))

(cl:ensure-generic-function 'option-val :lambda-list '(m))
(cl:defmethod option-val ((m <valueMatrix>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tiago_opencv_tutorial-msg:option-val is deprecated.  Use tiago_opencv_tutorial-msg:option instead.")
  (option m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <valueMatrix>) ostream)
  "Serializes a message object of type '<valueMatrix>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'tick) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'option))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'option))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <valueMatrix>) istream)
  "Deserializes a message object of type '<valueMatrix>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'value) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'tick) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'option) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'option) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<valueMatrix>)))
  "Returns string type for a message object of type '<valueMatrix>"
  "tiago_opencv_tutorial/valueMatrix")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'valueMatrix)))
  "Returns string type for a message object of type 'valueMatrix"
  "tiago_opencv_tutorial/valueMatrix")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<valueMatrix>)))
  "Returns md5sum for a message object of type '<valueMatrix>"
  "bc1cd923f6f816fbd3a3ec5219a648ae")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'valueMatrix)))
  "Returns md5sum for a message object of type 'valueMatrix"
  "bc1cd923f6f816fbd3a3ec5219a648ae")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<valueMatrix>)))
  "Returns full string definition for message of type '<valueMatrix>"
  (cl:format cl:nil "Header header~%~%float32 value~%~%bool tick~%~%string option~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'valueMatrix)))
  "Returns full string definition for message of type 'valueMatrix"
  (cl:format cl:nil "Header header~%~%float32 value~%~%bool tick~%~%string option~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <valueMatrix>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     1
     4 (cl:length (cl:slot-value msg 'option))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <valueMatrix>))
  "Converts a ROS message object to a list"
  (cl:list 'valueMatrix
    (cl:cons ':header (header msg))
    (cl:cons ':value (value msg))
    (cl:cons ':tick (tick msg))
    (cl:cons ':option (option msg))
))

; Auto-generated. Do not edit!


(cl:in-package play_motion_msgs-msg)


;//! \htmlinclude PlayMotionResult.msg.html

(cl:defclass <PlayMotionResult> (roslisp-msg-protocol:ros-message)
  ((error_code
    :reader error_code
    :initarg :error_code
    :type cl:integer
    :initform 0)
   (error_string
    :reader error_string
    :initarg :error_string
    :type cl:string
    :initform ""))
)

(cl:defclass PlayMotionResult (<PlayMotionResult>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PlayMotionResult>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PlayMotionResult)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name play_motion_msgs-msg:<PlayMotionResult> is deprecated: use play_motion_msgs-msg:PlayMotionResult instead.")))

(cl:ensure-generic-function 'error_code-val :lambda-list '(m))
(cl:defmethod error_code-val ((m <PlayMotionResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-msg:error_code-val is deprecated.  Use play_motion_msgs-msg:error_code instead.")
  (error_code m))

(cl:ensure-generic-function 'error_string-val :lambda-list '(m))
(cl:defmethod error_string-val ((m <PlayMotionResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-msg:error_string-val is deprecated.  Use play_motion_msgs-msg:error_string instead.")
  (error_string m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<PlayMotionResult>)))
    "Constants for message type '<PlayMotionResult>"
  '((:SUCCEEDED . 1)
    (:MOTION_NOT_FOUND . -1)
    (:CONTROLLER_BUSY . -3)
    (:MISSING_CONTROLLER . -4)
    (:TRAJECTORY_ERROR . -5)
    (:GOAL_NOT_REACHED . -6)
    (:PLANNER_OFFLINE . -7)
    (:NO_PLAN_FOUND . -8)
    (:OTHER_ERROR . -42))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'PlayMotionResult)))
    "Constants for message type 'PlayMotionResult"
  '((:SUCCEEDED . 1)
    (:MOTION_NOT_FOUND . -1)
    (:CONTROLLER_BUSY . -3)
    (:MISSING_CONTROLLER . -4)
    (:TRAJECTORY_ERROR . -5)
    (:GOAL_NOT_REACHED . -6)
    (:PLANNER_OFFLINE . -7)
    (:NO_PLAN_FOUND . -8)
    (:OTHER_ERROR . -42))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PlayMotionResult>) ostream)
  "Serializes a message object of type '<PlayMotionResult>"
  (cl:let* ((signed (cl:slot-value msg 'error_code)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'error_string))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'error_string))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PlayMotionResult>) istream)
  "Deserializes a message object of type '<PlayMotionResult>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'error_code) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'error_string) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'error_string) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PlayMotionResult>)))
  "Returns string type for a message object of type '<PlayMotionResult>"
  "play_motion_msgs/PlayMotionResult")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PlayMotionResult)))
  "Returns string type for a message object of type 'PlayMotionResult"
  "play_motion_msgs/PlayMotionResult")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PlayMotionResult>)))
  "Returns md5sum for a message object of type '<PlayMotionResult>"
  "5560cb9a7211cb4b5a730f6f07ec124b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PlayMotionResult)))
  "Returns md5sum for a message object of type 'PlayMotionResult"
  "5560cb9a7211cb4b5a730f6f07ec124b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PlayMotionResult>)))
  "Returns full string definition for message of type '<PlayMotionResult>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%int32 error_code~%int32 SUCCEEDED             = 1~%int32 MOTION_NOT_FOUND      = -1~%# controller error codes~%int32 CONTROLLER_BUSY       = -3~%int32 MISSING_CONTROLLER    = -4~%int32 TRAJECTORY_ERROR      = -5~%int32 GOAL_NOT_REACHED      = -6~%# planner error codes~%int32 PLANNER_OFFLINE       = -7~%int32 NO_PLAN_FOUND         = -8~%#other~%int32 OTHER_ERROR           = -42~%~%string error_string~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PlayMotionResult)))
  "Returns full string definition for message of type 'PlayMotionResult"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%int32 error_code~%int32 SUCCEEDED             = 1~%int32 MOTION_NOT_FOUND      = -1~%# controller error codes~%int32 CONTROLLER_BUSY       = -3~%int32 MISSING_CONTROLLER    = -4~%int32 TRAJECTORY_ERROR      = -5~%int32 GOAL_NOT_REACHED      = -6~%# planner error codes~%int32 PLANNER_OFFLINE       = -7~%int32 NO_PLAN_FOUND         = -8~%#other~%int32 OTHER_ERROR           = -42~%~%string error_string~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PlayMotionResult>))
  (cl:+ 0
     4
     4 (cl:length (cl:slot-value msg 'error_string))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PlayMotionResult>))
  "Converts a ROS message object to a list"
  (cl:list 'PlayMotionResult
    (cl:cons ':error_code (error_code msg))
    (cl:cons ':error_string (error_string msg))
))

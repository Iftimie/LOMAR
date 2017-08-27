; Auto-generated. Do not edit!


(cl:in-package play_motion_msgs-msg)


;//! \htmlinclude PlayMotionGoal.msg.html

(cl:defclass <PlayMotionGoal> (roslisp-msg-protocol:ros-message)
  ((motion_name
    :reader motion_name
    :initarg :motion_name
    :type cl:string
    :initform "")
   (skip_planning
    :reader skip_planning
    :initarg :skip_planning
    :type cl:boolean
    :initform cl:nil)
   (priority
    :reader priority
    :initarg :priority
    :type cl:integer
    :initform 0))
)

(cl:defclass PlayMotionGoal (<PlayMotionGoal>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PlayMotionGoal>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PlayMotionGoal)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name play_motion_msgs-msg:<PlayMotionGoal> is deprecated: use play_motion_msgs-msg:PlayMotionGoal instead.")))

(cl:ensure-generic-function 'motion_name-val :lambda-list '(m))
(cl:defmethod motion_name-val ((m <PlayMotionGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-msg:motion_name-val is deprecated.  Use play_motion_msgs-msg:motion_name instead.")
  (motion_name m))

(cl:ensure-generic-function 'skip_planning-val :lambda-list '(m))
(cl:defmethod skip_planning-val ((m <PlayMotionGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-msg:skip_planning-val is deprecated.  Use play_motion_msgs-msg:skip_planning instead.")
  (skip_planning m))

(cl:ensure-generic-function 'priority-val :lambda-list '(m))
(cl:defmethod priority-val ((m <PlayMotionGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-msg:priority-val is deprecated.  Use play_motion_msgs-msg:priority instead.")
  (priority m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PlayMotionGoal>) ostream)
  "Serializes a message object of type '<PlayMotionGoal>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'motion_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'motion_name))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'skip_planning) 1 0)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'priority)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PlayMotionGoal>) istream)
  "Deserializes a message object of type '<PlayMotionGoal>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'motion_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'motion_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'skip_planning) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'priority) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PlayMotionGoal>)))
  "Returns string type for a message object of type '<PlayMotionGoal>"
  "play_motion_msgs/PlayMotionGoal")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PlayMotionGoal)))
  "Returns string type for a message object of type 'PlayMotionGoal"
  "play_motion_msgs/PlayMotionGoal")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PlayMotionGoal>)))
  "Returns md5sum for a message object of type '<PlayMotionGoal>"
  "803175b562438759e071140087987d02")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PlayMotionGoal)))
  "Returns md5sum for a message object of type 'PlayMotionGoal"
  "803175b562438759e071140087987d02")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PlayMotionGoal>)))
  "Returns full string definition for message of type '<PlayMotionGoal>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%string motion_name~%bool skip_planning~%int32 priority~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PlayMotionGoal)))
  "Returns full string definition for message of type 'PlayMotionGoal"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%string motion_name~%bool skip_planning~%int32 priority~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PlayMotionGoal>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'motion_name))
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PlayMotionGoal>))
  "Converts a ROS message object to a list"
  (cl:list 'PlayMotionGoal
    (cl:cons ':motion_name (motion_name msg))
    (cl:cons ':skip_planning (skip_planning msg))
    (cl:cons ':priority (priority msg))
))

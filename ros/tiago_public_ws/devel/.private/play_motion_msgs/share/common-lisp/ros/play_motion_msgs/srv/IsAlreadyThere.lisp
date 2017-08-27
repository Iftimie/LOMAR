; Auto-generated. Do not edit!


(cl:in-package play_motion_msgs-srv)


;//! \htmlinclude IsAlreadyThere-request.msg.html

(cl:defclass <IsAlreadyThere-request> (roslisp-msg-protocol:ros-message)
  ((motion_name
    :reader motion_name
    :initarg :motion_name
    :type cl:string
    :initform "")
   (tolerance
    :reader tolerance
    :initarg :tolerance
    :type cl:float
    :initform 0.0))
)

(cl:defclass IsAlreadyThere-request (<IsAlreadyThere-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <IsAlreadyThere-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'IsAlreadyThere-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name play_motion_msgs-srv:<IsAlreadyThere-request> is deprecated: use play_motion_msgs-srv:IsAlreadyThere-request instead.")))

(cl:ensure-generic-function 'motion_name-val :lambda-list '(m))
(cl:defmethod motion_name-val ((m <IsAlreadyThere-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-srv:motion_name-val is deprecated.  Use play_motion_msgs-srv:motion_name instead.")
  (motion_name m))

(cl:ensure-generic-function 'tolerance-val :lambda-list '(m))
(cl:defmethod tolerance-val ((m <IsAlreadyThere-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-srv:tolerance-val is deprecated.  Use play_motion_msgs-srv:tolerance instead.")
  (tolerance m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <IsAlreadyThere-request>) ostream)
  "Serializes a message object of type '<IsAlreadyThere-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'motion_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'motion_name))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'tolerance))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <IsAlreadyThere-request>) istream)
  "Deserializes a message object of type '<IsAlreadyThere-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'motion_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'motion_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'tolerance) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<IsAlreadyThere-request>)))
  "Returns string type for a service object of type '<IsAlreadyThere-request>"
  "play_motion_msgs/IsAlreadyThereRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IsAlreadyThere-request)))
  "Returns string type for a service object of type 'IsAlreadyThere-request"
  "play_motion_msgs/IsAlreadyThereRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<IsAlreadyThere-request>)))
  "Returns md5sum for a message object of type '<IsAlreadyThere-request>"
  "7235124945b0db7b8593d4de73ead738")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'IsAlreadyThere-request)))
  "Returns md5sum for a message object of type 'IsAlreadyThere-request"
  "7235124945b0db7b8593d4de73ead738")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<IsAlreadyThere-request>)))
  "Returns full string definition for message of type '<IsAlreadyThere-request>"
  (cl:format cl:nil "~%~%~%~%~%~%~%~%~%string motion_name~%float32 tolerance~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'IsAlreadyThere-request)))
  "Returns full string definition for message of type 'IsAlreadyThere-request"
  (cl:format cl:nil "~%~%~%~%~%~%~%~%~%string motion_name~%float32 tolerance~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <IsAlreadyThere-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'motion_name))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <IsAlreadyThere-request>))
  "Converts a ROS message object to a list"
  (cl:list 'IsAlreadyThere-request
    (cl:cons ':motion_name (motion_name msg))
    (cl:cons ':tolerance (tolerance msg))
))
;//! \htmlinclude IsAlreadyThere-response.msg.html

(cl:defclass <IsAlreadyThere-response> (roslisp-msg-protocol:ros-message)
  ((already_there
    :reader already_there
    :initarg :already_there
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass IsAlreadyThere-response (<IsAlreadyThere-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <IsAlreadyThere-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'IsAlreadyThere-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name play_motion_msgs-srv:<IsAlreadyThere-response> is deprecated: use play_motion_msgs-srv:IsAlreadyThere-response instead.")))

(cl:ensure-generic-function 'already_there-val :lambda-list '(m))
(cl:defmethod already_there-val ((m <IsAlreadyThere-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader play_motion_msgs-srv:already_there-val is deprecated.  Use play_motion_msgs-srv:already_there instead.")
  (already_there m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <IsAlreadyThere-response>) ostream)
  "Serializes a message object of type '<IsAlreadyThere-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'already_there) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <IsAlreadyThere-response>) istream)
  "Deserializes a message object of type '<IsAlreadyThere-response>"
    (cl:setf (cl:slot-value msg 'already_there) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<IsAlreadyThere-response>)))
  "Returns string type for a service object of type '<IsAlreadyThere-response>"
  "play_motion_msgs/IsAlreadyThereResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IsAlreadyThere-response)))
  "Returns string type for a service object of type 'IsAlreadyThere-response"
  "play_motion_msgs/IsAlreadyThereResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<IsAlreadyThere-response>)))
  "Returns md5sum for a message object of type '<IsAlreadyThere-response>"
  "7235124945b0db7b8593d4de73ead738")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'IsAlreadyThere-response)))
  "Returns md5sum for a message object of type 'IsAlreadyThere-response"
  "7235124945b0db7b8593d4de73ead738")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<IsAlreadyThere-response>)))
  "Returns full string definition for message of type '<IsAlreadyThere-response>"
  (cl:format cl:nil "bool already_there~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'IsAlreadyThere-response)))
  "Returns full string definition for message of type 'IsAlreadyThere-response"
  (cl:format cl:nil "bool already_there~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <IsAlreadyThere-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <IsAlreadyThere-response>))
  "Converts a ROS message object to a list"
  (cl:list 'IsAlreadyThere-response
    (cl:cons ':already_there (already_there msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'IsAlreadyThere)))
  'IsAlreadyThere-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'IsAlreadyThere)))
  'IsAlreadyThere-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IsAlreadyThere)))
  "Returns string type for a service object of type '<IsAlreadyThere>"
  "play_motion_msgs/IsAlreadyThere")
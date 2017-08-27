; Auto-generated. Do not edit!


(cl:in-package tf_lookup-msg)


;//! \htmlinclude Subscription.msg.html

(cl:defclass <Subscription> (roslisp-msg-protocol:ros-message)
  ((target
    :reader target
    :initarg :target
    :type cl:string
    :initform "")
   (source
    :reader source
    :initarg :source
    :type cl:string
    :initform ""))
)

(cl:defclass Subscription (<Subscription>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Subscription>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Subscription)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tf_lookup-msg:<Subscription> is deprecated: use tf_lookup-msg:Subscription instead.")))

(cl:ensure-generic-function 'target-val :lambda-list '(m))
(cl:defmethod target-val ((m <Subscription>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:target-val is deprecated.  Use tf_lookup-msg:target instead.")
  (target m))

(cl:ensure-generic-function 'source-val :lambda-list '(m))
(cl:defmethod source-val ((m <Subscription>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:source-val is deprecated.  Use tf_lookup-msg:source instead.")
  (source m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Subscription>) ostream)
  "Serializes a message object of type '<Subscription>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'target))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'target))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'source))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'source))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Subscription>) istream)
  "Deserializes a message object of type '<Subscription>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'target) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'target) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'source) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'source) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Subscription>)))
  "Returns string type for a message object of type '<Subscription>"
  "tf_lookup/Subscription")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Subscription)))
  "Returns string type for a message object of type 'Subscription"
  "tf_lookup/Subscription")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Subscription>)))
  "Returns md5sum for a message object of type '<Subscription>"
  "2f26b645cf5988fb447dd948d42f673d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Subscription)))
  "Returns md5sum for a message object of type 'Subscription"
  "2f26b645cf5988fb447dd948d42f673d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Subscription>)))
  "Returns full string definition for message of type '<Subscription>"
  (cl:format cl:nil "string target~%string source~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Subscription)))
  "Returns full string definition for message of type 'Subscription"
  (cl:format cl:nil "string target~%string source~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Subscription>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'target))
     4 (cl:length (cl:slot-value msg 'source))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Subscription>))
  "Converts a ROS message object to a list"
  (cl:list 'Subscription
    (cl:cons ':target (target msg))
    (cl:cons ':source (source msg))
))

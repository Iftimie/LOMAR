; Auto-generated. Do not edit!


(cl:in-package tf_lookup-msg)


;//! \htmlinclude TfLookupGoal.msg.html

(cl:defclass <TfLookupGoal> (roslisp-msg-protocol:ros-message)
  ((target_frame
    :reader target_frame
    :initarg :target_frame
    :type cl:string
    :initform "")
   (source_frame
    :reader source_frame
    :initarg :source_frame
    :type cl:string
    :initform "")
   (transform_time
    :reader transform_time
    :initarg :transform_time
    :type cl:real
    :initform 0))
)

(cl:defclass TfLookupGoal (<TfLookupGoal>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TfLookupGoal>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TfLookupGoal)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tf_lookup-msg:<TfLookupGoal> is deprecated: use tf_lookup-msg:TfLookupGoal instead.")))

(cl:ensure-generic-function 'target_frame-val :lambda-list '(m))
(cl:defmethod target_frame-val ((m <TfLookupGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:target_frame-val is deprecated.  Use tf_lookup-msg:target_frame instead.")
  (target_frame m))

(cl:ensure-generic-function 'source_frame-val :lambda-list '(m))
(cl:defmethod source_frame-val ((m <TfLookupGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:source_frame-val is deprecated.  Use tf_lookup-msg:source_frame instead.")
  (source_frame m))

(cl:ensure-generic-function 'transform_time-val :lambda-list '(m))
(cl:defmethod transform_time-val ((m <TfLookupGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:transform_time-val is deprecated.  Use tf_lookup-msg:transform_time instead.")
  (transform_time m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TfLookupGoal>) ostream)
  "Serializes a message object of type '<TfLookupGoal>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'target_frame))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'target_frame))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'source_frame))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'source_frame))
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'transform_time)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'transform_time) (cl:floor (cl:slot-value msg 'transform_time)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TfLookupGoal>) istream)
  "Deserializes a message object of type '<TfLookupGoal>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'target_frame) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'target_frame) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'source_frame) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'source_frame) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'transform_time) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TfLookupGoal>)))
  "Returns string type for a message object of type '<TfLookupGoal>"
  "tf_lookup/TfLookupGoal")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TfLookupGoal)))
  "Returns string type for a message object of type 'TfLookupGoal"
  "tf_lookup/TfLookupGoal")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TfLookupGoal>)))
  "Returns md5sum for a message object of type '<TfLookupGoal>"
  "bb9d983758e61f286b43546ac9c0b080")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TfLookupGoal)))
  "Returns md5sum for a message object of type 'TfLookupGoal"
  "bb9d983758e61f286b43546ac9c0b080")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TfLookupGoal>)))
  "Returns full string definition for message of type '<TfLookupGoal>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# Define the goal~%string target_frame #The frame to which data should be transformed~%string source_frame #The frame where the data originated~%time transform_time~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TfLookupGoal)))
  "Returns full string definition for message of type 'TfLookupGoal"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# Define the goal~%string target_frame #The frame to which data should be transformed~%string source_frame #The frame where the data originated~%time transform_time~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TfLookupGoal>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'target_frame))
     4 (cl:length (cl:slot-value msg 'source_frame))
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TfLookupGoal>))
  "Converts a ROS message object to a list"
  (cl:list 'TfLookupGoal
    (cl:cons ':target_frame (target_frame msg))
    (cl:cons ':source_frame (source_frame msg))
    (cl:cons ':transform_time (transform_time msg))
))

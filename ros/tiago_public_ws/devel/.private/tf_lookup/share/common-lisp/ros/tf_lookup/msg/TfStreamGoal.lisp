; Auto-generated. Do not edit!


(cl:in-package tf_lookup-msg)


;//! \htmlinclude TfStreamGoal.msg.html

(cl:defclass <TfStreamGoal> (roslisp-msg-protocol:ros-message)
  ((transforms
    :reader transforms
    :initarg :transforms
    :type (cl:vector tf_lookup-msg:Subscription)
   :initform (cl:make-array 0 :element-type 'tf_lookup-msg:Subscription :initial-element (cl:make-instance 'tf_lookup-msg:Subscription)))
   (subscription_id
    :reader subscription_id
    :initarg :subscription_id
    :type cl:string
    :initform "")
   (update
    :reader update
    :initarg :update
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass TfStreamGoal (<TfStreamGoal>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TfStreamGoal>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TfStreamGoal)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tf_lookup-msg:<TfStreamGoal> is deprecated: use tf_lookup-msg:TfStreamGoal instead.")))

(cl:ensure-generic-function 'transforms-val :lambda-list '(m))
(cl:defmethod transforms-val ((m <TfStreamGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:transforms-val is deprecated.  Use tf_lookup-msg:transforms instead.")
  (transforms m))

(cl:ensure-generic-function 'subscription_id-val :lambda-list '(m))
(cl:defmethod subscription_id-val ((m <TfStreamGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:subscription_id-val is deprecated.  Use tf_lookup-msg:subscription_id instead.")
  (subscription_id m))

(cl:ensure-generic-function 'update-val :lambda-list '(m))
(cl:defmethod update-val ((m <TfStreamGoal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tf_lookup-msg:update-val is deprecated.  Use tf_lookup-msg:update instead.")
  (update m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TfStreamGoal>) ostream)
  "Serializes a message object of type '<TfStreamGoal>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'transforms))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'transforms))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'subscription_id))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'subscription_id))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TfStreamGoal>) istream)
  "Deserializes a message object of type '<TfStreamGoal>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'transforms) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'transforms)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tf_lookup-msg:Subscription))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'subscription_id) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'subscription_id) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'update) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TfStreamGoal>)))
  "Returns string type for a message object of type '<TfStreamGoal>"
  "tf_lookup/TfStreamGoal")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TfStreamGoal)))
  "Returns string type for a message object of type 'TfStreamGoal"
  "tf_lookup/TfStreamGoal")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TfStreamGoal>)))
  "Returns md5sum for a message object of type '<TfStreamGoal>"
  "e00b5ec9adf5765d948ec802ab721a4a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TfStreamGoal)))
  "Returns md5sum for a message object of type 'TfStreamGoal"
  "e00b5ec9adf5765d948ec802ab721a4a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TfStreamGoal>)))
  "Returns full string definition for message of type '<TfStreamGoal>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# Define the goal~%Subscription[] transforms~%string subscription_id~%bool update~%~%================================================================================~%MSG: tf_lookup/Subscription~%string target~%string source~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TfStreamGoal)))
  "Returns full string definition for message of type 'TfStreamGoal"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# Define the goal~%Subscription[] transforms~%string subscription_id~%bool update~%~%================================================================================~%MSG: tf_lookup/Subscription~%string target~%string source~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TfStreamGoal>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'transforms) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:length (cl:slot-value msg 'subscription_id))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TfStreamGoal>))
  "Converts a ROS message object to a list"
  (cl:list 'TfStreamGoal
    (cl:cons ':transforms (transforms msg))
    (cl:cons ':subscription_id (subscription_id msg))
    (cl:cons ':update (update msg))
))

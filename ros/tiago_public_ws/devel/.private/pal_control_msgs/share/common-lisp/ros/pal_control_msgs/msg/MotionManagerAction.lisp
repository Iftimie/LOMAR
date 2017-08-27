; Auto-generated. Do not edit!


(cl:in-package pal_control_msgs-msg)


;//! \htmlinclude MotionManagerAction.msg.html

(cl:defclass <MotionManagerAction> (roslisp-msg-protocol:ros-message)
  ((action_goal
    :reader action_goal
    :initarg :action_goal
    :type pal_control_msgs-msg:MotionManagerActionGoal
    :initform (cl:make-instance 'pal_control_msgs-msg:MotionManagerActionGoal))
   (action_result
    :reader action_result
    :initarg :action_result
    :type pal_control_msgs-msg:MotionManagerActionResult
    :initform (cl:make-instance 'pal_control_msgs-msg:MotionManagerActionResult))
   (action_feedback
    :reader action_feedback
    :initarg :action_feedback
    :type pal_control_msgs-msg:MotionManagerActionFeedback
    :initform (cl:make-instance 'pal_control_msgs-msg:MotionManagerActionFeedback)))
)

(cl:defclass MotionManagerAction (<MotionManagerAction>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MotionManagerAction>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MotionManagerAction)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name pal_control_msgs-msg:<MotionManagerAction> is deprecated: use pal_control_msgs-msg:MotionManagerAction instead.")))

(cl:ensure-generic-function 'action_goal-val :lambda-list '(m))
(cl:defmethod action_goal-val ((m <MotionManagerAction>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pal_control_msgs-msg:action_goal-val is deprecated.  Use pal_control_msgs-msg:action_goal instead.")
  (action_goal m))

(cl:ensure-generic-function 'action_result-val :lambda-list '(m))
(cl:defmethod action_result-val ((m <MotionManagerAction>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pal_control_msgs-msg:action_result-val is deprecated.  Use pal_control_msgs-msg:action_result instead.")
  (action_result m))

(cl:ensure-generic-function 'action_feedback-val :lambda-list '(m))
(cl:defmethod action_feedback-val ((m <MotionManagerAction>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pal_control_msgs-msg:action_feedback-val is deprecated.  Use pal_control_msgs-msg:action_feedback instead.")
  (action_feedback m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MotionManagerAction>) ostream)
  "Serializes a message object of type '<MotionManagerAction>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'action_goal) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'action_result) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'action_feedback) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MotionManagerAction>) istream)
  "Deserializes a message object of type '<MotionManagerAction>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'action_goal) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'action_result) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'action_feedback) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MotionManagerAction>)))
  "Returns string type for a message object of type '<MotionManagerAction>"
  "pal_control_msgs/MotionManagerAction")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MotionManagerAction)))
  "Returns string type for a message object of type 'MotionManagerAction"
  "pal_control_msgs/MotionManagerAction")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MotionManagerAction>)))
  "Returns md5sum for a message object of type '<MotionManagerAction>"
  "42689d3bf9c1135e4da2202787f92626")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MotionManagerAction)))
  "Returns md5sum for a message object of type 'MotionManagerAction"
  "42689d3bf9c1135e4da2202787f92626")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MotionManagerAction>)))
  "Returns full string definition for message of type '<MotionManagerAction>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%MotionManagerActionGoal action_goal~%MotionManagerActionResult action_result~%MotionManagerActionFeedback action_feedback~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerActionGoal~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%Header header~%actionlib_msgs/GoalID goal_id~%MotionManagerGoal goal~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: actionlib_msgs/GoalID~%# The stamp should store the time at which this goal was requested.~%# It is used by an action server when it tries to preempt all~%# goals that were requested before a certain time~%time stamp~%~%# The id provides a way to associate feedback and~%# result message with specific goal requests. The id~%# specified must be unique.~%string id~%~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerGoal~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# Path to XML file containing motions for the robot~%string filename~%~%# True if a collision-free approach motion and trajectory validation are to be performed.~%# If set to true but an approach motion is not required, it will not be computed.~%bool plan~%~%#True if safety around the robot must be checked using sensors such as the sonars and lasers~%bool checkSafety~%~%#True if the motion must be repeated until a new goal has been received~%bool repeat~%~%#priority of the motion, 0 is no priority, 100 is max priority~%uint8 priority~%~%#Specifies how long in miliseconds should the goal wait before forcing an execution. If a movement is being executed when the goal is received, it will wait the specified time or until the movement finishes to execute it.~%# -1 Means wait forever until the previous movement has finished~%int32 queueTimeout ~%~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerActionResult~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%Header header~%actionlib_msgs/GoalStatus status~%MotionManagerResult result~%~%================================================================================~%MSG: actionlib_msgs/GoalStatus~%GoalID goal_id~%uint8 status~%uint8 PENDING         = 0   # The goal has yet to be processed by the action server~%uint8 ACTIVE          = 1   # The goal is currently being processed by the action server~%uint8 PREEMPTED       = 2   # The goal received a cancel request after it started executing~%                            #   and has since completed its execution (Terminal State)~%uint8 SUCCEEDED       = 3   # The goal was achieved successfully by the action server (Terminal State)~%uint8 ABORTED         = 4   # The goal was aborted during execution by the action server due~%                            #    to some failure (Terminal State)~%uint8 REJECTED        = 5   # The goal was rejected by the action server without being processed,~%                            #    because the goal was unattainable or invalid (Terminal State)~%uint8 PREEMPTING      = 6   # The goal received a cancel request after it started executing~%                            #    and has not yet completed execution~%uint8 RECALLING       = 7   # The goal received a cancel request before it started executing,~%                            #    but the action server has not yet confirmed that the goal is canceled~%uint8 RECALLED        = 8   # The goal received a cancel request before it started executing~%                            #    and was successfully cancelled (Terminal State)~%uint8 LOST            = 9   # An action client can determine that a goal is LOST. This should not be~%                            #    sent over the wire by an action server~%~%#Allow for the user to associate a string with GoalStatus for debugging~%string text~%~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerResult~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%#Message in result, can contain information if goal failed~%string message~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerActionFeedback~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%Header header~%actionlib_msgs/GoalStatus status~%MotionManagerFeedback feedback~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerFeedback~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# feedback message~%# no feedback for the moment. could be progress, or final position~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MotionManagerAction)))
  "Returns full string definition for message of type 'MotionManagerAction"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%MotionManagerActionGoal action_goal~%MotionManagerActionResult action_result~%MotionManagerActionFeedback action_feedback~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerActionGoal~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%Header header~%actionlib_msgs/GoalID goal_id~%MotionManagerGoal goal~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: actionlib_msgs/GoalID~%# The stamp should store the time at which this goal was requested.~%# It is used by an action server when it tries to preempt all~%# goals that were requested before a certain time~%time stamp~%~%# The id provides a way to associate feedback and~%# result message with specific goal requests. The id~%# specified must be unique.~%string id~%~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerGoal~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# Path to XML file containing motions for the robot~%string filename~%~%# True if a collision-free approach motion and trajectory validation are to be performed.~%# If set to true but an approach motion is not required, it will not be computed.~%bool plan~%~%#True if safety around the robot must be checked using sensors such as the sonars and lasers~%bool checkSafety~%~%#True if the motion must be repeated until a new goal has been received~%bool repeat~%~%#priority of the motion, 0 is no priority, 100 is max priority~%uint8 priority~%~%#Specifies how long in miliseconds should the goal wait before forcing an execution. If a movement is being executed when the goal is received, it will wait the specified time or until the movement finishes to execute it.~%# -1 Means wait forever until the previous movement has finished~%int32 queueTimeout ~%~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerActionResult~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%Header header~%actionlib_msgs/GoalStatus status~%MotionManagerResult result~%~%================================================================================~%MSG: actionlib_msgs/GoalStatus~%GoalID goal_id~%uint8 status~%uint8 PENDING         = 0   # The goal has yet to be processed by the action server~%uint8 ACTIVE          = 1   # The goal is currently being processed by the action server~%uint8 PREEMPTED       = 2   # The goal received a cancel request after it started executing~%                            #   and has since completed its execution (Terminal State)~%uint8 SUCCEEDED       = 3   # The goal was achieved successfully by the action server (Terminal State)~%uint8 ABORTED         = 4   # The goal was aborted during execution by the action server due~%                            #    to some failure (Terminal State)~%uint8 REJECTED        = 5   # The goal was rejected by the action server without being processed,~%                            #    because the goal was unattainable or invalid (Terminal State)~%uint8 PREEMPTING      = 6   # The goal received a cancel request after it started executing~%                            #    and has not yet completed execution~%uint8 RECALLING       = 7   # The goal received a cancel request before it started executing,~%                            #    but the action server has not yet confirmed that the goal is canceled~%uint8 RECALLED        = 8   # The goal received a cancel request before it started executing~%                            #    and was successfully cancelled (Terminal State)~%uint8 LOST            = 9   # An action client can determine that a goal is LOST. This should not be~%                            #    sent over the wire by an action server~%~%#Allow for the user to associate a string with GoalStatus for debugging~%string text~%~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerResult~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%#Message in result, can contain information if goal failed~%string message~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerActionFeedback~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%~%Header header~%actionlib_msgs/GoalStatus status~%MotionManagerFeedback feedback~%~%================================================================================~%MSG: pal_control_msgs/MotionManagerFeedback~%# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%# feedback message~%# no feedback for the moment. could be progress, or final position~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MotionManagerAction>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'action_goal))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'action_result))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'action_feedback))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MotionManagerAction>))
  "Converts a ROS message object to a list"
  (cl:list 'MotionManagerAction
    (cl:cons ':action_goal (action_goal msg))
    (cl:cons ':action_result (action_result msg))
    (cl:cons ':action_feedback (action_feedback msg))
))

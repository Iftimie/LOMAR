
(cl:in-package :asdf)

(defsystem "play_motion_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :actionlib_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "MotionInfo" :depends-on ("_package_MotionInfo"))
    (:file "_package_MotionInfo" :depends-on ("_package"))
    (:file "PlayMotionFeedback" :depends-on ("_package_PlayMotionFeedback"))
    (:file "_package_PlayMotionFeedback" :depends-on ("_package"))
    (:file "PlayMotionActionFeedback" :depends-on ("_package_PlayMotionActionFeedback"))
    (:file "_package_PlayMotionActionFeedback" :depends-on ("_package"))
    (:file "PlayMotionActionResult" :depends-on ("_package_PlayMotionActionResult"))
    (:file "_package_PlayMotionActionResult" :depends-on ("_package"))
    (:file "PlayMotionGoal" :depends-on ("_package_PlayMotionGoal"))
    (:file "_package_PlayMotionGoal" :depends-on ("_package"))
    (:file "PlayMotionActionGoal" :depends-on ("_package_PlayMotionActionGoal"))
    (:file "_package_PlayMotionActionGoal" :depends-on ("_package"))
    (:file "PlayMotionResult" :depends-on ("_package_PlayMotionResult"))
    (:file "_package_PlayMotionResult" :depends-on ("_package"))
    (:file "PlayMotionAction" :depends-on ("_package_PlayMotionAction"))
    (:file "_package_PlayMotionAction" :depends-on ("_package"))
  ))
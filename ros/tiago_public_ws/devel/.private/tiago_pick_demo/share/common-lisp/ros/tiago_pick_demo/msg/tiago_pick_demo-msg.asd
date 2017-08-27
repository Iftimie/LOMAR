
(cl:in-package :asdf)

(defsystem "tiago_pick_demo-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :actionlib_msgs-msg
               :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "PickUpPoseGoal" :depends-on ("_package_PickUpPoseGoal"))
    (:file "_package_PickUpPoseGoal" :depends-on ("_package"))
    (:file "PickUpPoseActionGoal" :depends-on ("_package_PickUpPoseActionGoal"))
    (:file "_package_PickUpPoseActionGoal" :depends-on ("_package"))
    (:file "PickUpPoseActionFeedback" :depends-on ("_package_PickUpPoseActionFeedback"))
    (:file "_package_PickUpPoseActionFeedback" :depends-on ("_package"))
    (:file "PickUpPoseFeedback" :depends-on ("_package_PickUpPoseFeedback"))
    (:file "_package_PickUpPoseFeedback" :depends-on ("_package"))
    (:file "PickUpPoseResult" :depends-on ("_package_PickUpPoseResult"))
    (:file "_package_PickUpPoseResult" :depends-on ("_package"))
    (:file "PickUpPoseActionResult" :depends-on ("_package_PickUpPoseActionResult"))
    (:file "_package_PickUpPoseActionResult" :depends-on ("_package"))
    (:file "PickUpPoseAction" :depends-on ("_package_PickUpPoseAction"))
    (:file "_package_PickUpPoseAction" :depends-on ("_package"))
  ))
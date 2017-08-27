
(cl:in-package :asdf)

(defsystem "play_motion_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :play_motion_msgs-msg
)
  :components ((:file "_package")
    (:file "ListMotions" :depends-on ("_package_ListMotions"))
    (:file "_package_ListMotions" :depends-on ("_package"))
    (:file "IsAlreadyThere" :depends-on ("_package_IsAlreadyThere"))
    (:file "_package_IsAlreadyThere" :depends-on ("_package"))
  ))
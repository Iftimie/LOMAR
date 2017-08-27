
(cl:in-package :asdf)

(defsystem "diff_drive_controller-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "WheelData" :depends-on ("_package_WheelData"))
    (:file "_package_WheelData" :depends-on ("_package"))
  ))
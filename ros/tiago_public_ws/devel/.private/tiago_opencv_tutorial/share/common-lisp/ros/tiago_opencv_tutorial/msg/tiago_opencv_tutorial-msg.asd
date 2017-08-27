
(cl:in-package :asdf)

(defsystem "tiago_opencv_tutorial-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "valueMatrix" :depends-on ("_package_valueMatrix"))
    (:file "_package_valueMatrix" :depends-on ("_package"))
  ))
# LOMAR
Intelligent indoor LOcalization and MApping based on a Robotic platform LOMAR

This repository contains the research on an Indoor localization system based on RSS (Received Signal Strenght) from AP(Acces Points) or WLAN routers. The ideea is that many buildings nowadays have tens or hundreds of routers inside them. All the signal traveling through the air can be used for localization. One way to do this is to triangulate the position knowing the coordinates of all routers. But not everybody has acces to all rooms in a building. Our approach intends to use a neural network that learns the positions in the building given the RSS vector.
To automate the process of data collection we have a robot that is able to generate the pair X (RSS vector) and Y (x y coordinates). The robot integrates multiple sensors in order to do accurate localization. The robot must navigate autonomously through the building, compute the location given its sensors and with a WIFI scanner it must collect the RSS vector.
The neural network has a regression head, not a classification head.
More details are in the paper.docx.

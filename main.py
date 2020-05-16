"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5 && reset
reset && python3 main.py -m models/project7.xml -i test_video.mp4 -l "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so" -o out.mp4

reset && python3 main.py -m models/project7.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so" -o out1.mp4

reset && python3 main.py -d CPU -m models/project7.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so" -pt 0.5 -iou 0.2 -md 1

reset && python3 main.py -d CPU -i resources/Pedestrian_Detect_2_1_1.mp4 -l "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so" -m models/project7.xml -pt 0.5 -iou 0.2 -md 1 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
"""


import os
import sys
import time
import socket
import json
import cv2
import argparse

import logging as log
import paho.mqtt.client as mqtt
import numpy as np

# from argparse import ArgumentParser
# from inference import Network
from utils import *

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

class_path='coco.names'
classes = load_class_names(class_path)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser("Run inference on an input video")
    
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    required.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    optional.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    optional.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    optional.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    optional.add_argument("-iou","--inter_over_union",type=float,default=0.2,help="Probabilty of allowing overlaying boxes to either stay or leave""(0.2 by default)")
    optional.add_argument("-o","--output",type=str,default=None,help="saves output to local machine where the whole network runs")
    optional.add_argument("-md","--mode",type=int,default=1,help="mode in which to run network (1) for async (0) for sync mode respectively")
    return parser

# CPU = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


def connect_mqtt():
    #     ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

# def get_corr_frame_shape(input_shape=None,frame=None):
#     """
#         This Function is Used to preprocess a single frame to appriopraite
#         Size and dimension
        
#         parameters:
#             agr1(inputshape) : This is the 4-dim input gotten from an instance of the IENetwork,
#             input shoulf be arrange in the format (batch-size, channel, height, width)
#             arg2(frame) : This is the image that is about to be passed to the Network for 
#                           Inference
                      
#         Example:
#             get_corr_frame_shape(input_shape=(1,3,416,416),frame=image)
#     """
#     resized_frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
#     resized_frame =  np.expand_dims(resized_frame.transpose((2,0,1))/255,axis=0)
#     return resized_frame

# def infer_img(inf_net=None,img_ref=None,iou=0.3,nms=0.0,out_serveronly=True,out_name=None):
#     """
#         This Function is Used to pass an image for inference. 
        
#         parameters:
#             agr1(inf_net) : This is an Instance of The Network Class, Which has the 
#                             architecture and weight loaded in the IENetwork and the
#                             Network into the IECore.
#             arg2(img_ref) : This is the path to the image on the local machine.
            
#             arg3(out_serveronly): This is a boolean value to indicate if the result 
#                             should be saved on the local machine also.
#             arg4(out_name): If out_serveronly is False this Value of this augument 
#                             will be the file name for the output.
                      
#         Example:
#             infer_img(inf_net=Network,img_ref="abc/def.jpg",out_serveronly=True,out_name=None)
#     """
#     ### TODO: Handle the input stream ###
#     input_shape = inf_net.get_input_shape() 
#     frame = cv2.imread(img_ref)
#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     resized_frame = get_corr_frame_shape(input_shape, frame)
#     frame_and_resframe = {}
#     #variable to indicate  the curret input id. the variable can fast forward.
#     current_input = 0 
#     #variable to indicate the current request that we need it get it's request
#     curr_need = 0 
    
#     frame_bgr = cv2.imread(img_ref)
#     frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
#     resized_frame = get_corr_frame_shape(input_shape, frame_rgb)
# #     frame_and_resframe = {}
# #     {current_input:{
# #                                         'frame': frame,
# #                                         'resized_frame':resized_frame}}
# #     current

# #     print(id_)
#     while True:
#         #send frame for inference
#         inf_net.exec_net_async(resized_frame,id_=current_input)
#         #increase to the next frame if available
# #         current_input += 1
        
#         ### Get the output of inference if ready
#         if inf_net.wait(curr_need) == 0:
#             #increse to the current frame result that needs to be gotten
#             print('checking',iou,nms)
#             result = inf_net.get_output(curr_need,iou,nms)

#             ### TODO: Update the frame to include detected bounding boxes
#             frame = plot_boxes(frame_bgr, result,classes, plot_labels=True)
            
#             # Write out the frame
#             cv2.imwrite(out_name,frame)
#             break
            
    
#         # Break if escape key pressed
#         if key_pressed == 27:
#             break

#     # Release the out writer, capture, and destroy any OpenCV windows
# #     out.release()
# #     cap.release()
#     cv2.destroyAllWindows()    
#     return

# def infer_vid(inf_net=None,vid_ref=None,out_serveronly=True,out_name=None):
#     ### TODO: Handle the input stream ###
#     input_shape = inf_net.get_input_shape()
    
#     ### TODO: Read from the video capture ###
#     img = cv2.imread()
#     ### TODO: Pre-process the image as needed ###

#     ### TODO: Start asynchronous inference for specified request ###

#     ### TODO: Wait for the result ###

#     ### TODO: Get the results of the inference request ###

#     ### TODO: Extract any desired stats from the results ###

#     ### TODO: Calculate and send relevant information on ###
#     ### current_count, total_count and duration to the MQTT server ###
#     ### Topic "person": keys of "count" and "total" ###
#     ### Topic "person/duration": key of "duration" ###

#     ### TODO: Send the frame to the FFMPEG server ###

#     ### TODO: Write an output image if `single_image_mode` ###
#     pass

# def infer_webcam(inf_net=None,out_serveronly=False,out_name=None):
#     ### TODO: Handle the input stream ###
#     input_shape = inf_net.get_input_shape()
#     pass

def infer_on_stream(args, server=None):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    input_ = args.input

    #check if server is available
    
    #Initialize tracker class which in turns Initialize IEcor and IENwtwork
    tracker = human_tracker(model=args.model,device=args.device, cpu_ext=args.cpu_extension,mqtt_server=server)
    
    #set parameters required for network to run
    tracker.prob_threshold = args.prob_threshold
    tracker.prob_iou = args.inter_over_union
    tracker.out_name = args.output
    print('checking from outside', tracker.out_name)
    tracker.out_serveronly = False if tracker.out_name else True
    tracker.classes = classes
    start = time.time()
    print('i got fmor outer ', input_)
    tracker.run(input_=input_,async_=args.mode)
    print('End Time is {}'.format(time.time()-start))
    
#     ### TODO: Load the model through `infer_network` ###
#     infer_network.load_model(model,device,cpu_ext)
    
        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###

        ### TODO: Start asynchronous inference for specified request ###

        ### TODO: Wait for the result ###

            ### TODO: Get the results of the inference request ###

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, server=client)


if __name__ == '__main__':
    main()

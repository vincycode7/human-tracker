import time,cv2,json,sys,warnings
import numpy as np
from inference import Network

def load_class_names(namesfile):
    
    # Create an empty list to hold the object classes
    class_names = []
    
    # Open the file containing the COCO object classes in read-only mode
    with open(namesfile, 'r') as fp:
        
        # The coco.names file contains only one object class per line.
        # Read the file line by line and save all the lines in a list.
        lines = fp.readlines()
    
    # Get the object class names
    for line in lines:
        
        # Make a copy of each line with any trailing whitespace removed
        line = line.rstrip()
        
        # Save the object class name into class_names
        class_names.append(line)
        
    return class_names

class human_tracker(object):
    def __init__(self, model,device,cpu_ext,mqtt_server):
        #required arguments
        self.model = model
        self.device = device
        self.cpu_ext = cpu_ext
        self.mqtt_client = mqtt_server
        
        #argument for other functions
        self.prob_threshold = None
        self.prob_iou = None
        self.input_ = None
        self._out_name = None
        self.out_serveronly = False
        self.classes = None
        self.frames = {}
        self.prev_boxes = {}
        self.curr_boxes = {}
#         self.cuur_out = []
        self.tolerate = {}
        self.total_detect = 0
        self.avg_time = 0
        self.async_ = None
        
        # Initialise the class
        self.infer_network = Network()
    
        ### TODO: Load the model through `infer_network` ###
        self.infer_network.load_model(self.model,self.device,self.cpu_ext)
        
        #check if server is available
        self.check_server()
        
    @property
    def out_name(self):
        return self._out_name
    
    @out_name.setter
    def out_name(self, value):
        if value:
            #Check if to also save result from network to machine running the network
            ext_file = ['jpg','png','jpeg'] if value.split('.')[1] in ['png','jpg','jpeg'] else ['mp4','mkv']
            if not self.out_serveronly:
                if len(value.split('.')) == 1:
                    self._out_name = value +ext_file[0]
                    return
                elif len(value.split('.')) == 2 and value.split('.')[1] in ext_file:
                    self._out_name = value
                    return
                elif len(value.split('.')) == 2 and value.split('.')[1] not in ext_file:
                    self._out_name = value.split('.')[0]+ext_file[0]
                    return
                else:
                    raise ValueError('There is something wrong with the file name you specified for output')
            else:
                self.out_serveronly = False
                self._out_name = None
        self._out_name = value
    
    def run(self, input_,async_=True):
        self.async_ = async_
        if self.async_==None:
            self.async_ = True
            msg = "Run mode is None changed to async"
            warnings.warn(msg, UserWarning)
        print('mode is {}'.format(self.async_))
        if not input_:
            raise ValueError('Input Required')
        if self.out_name and len(input_.split('.')) > 1 and len(self.out_name.split('.')) > 1:
            if input_.split('.')[1] != self.out_name.split('.')[1]:
                out_name = self.out_name.split('.')[0]+ '.'+input_.split('.')[1]
                msg = "Output have different extensions out with {} and input with {} new output is {}".format(self.out_name.split('.')[1],input_.split('.')[1],out_name)
                warnings.warn(msg, UserWarning)
                self.out_name = out_name
        if not self.classes:
            raise ValueError('classes Required')
            
        runner_ = self.runner(input_)
        self.input_ = input_
        
        ### TODO: Loop until stream is over ###
        self.infer_start = time.time()
        runner_()
        
        print('Done')   
        return
        
    
    def __call__(self, input_,async_=True):
        if not input_:
            ValueError('Input Required')
        self.run(input_,async_)
        return 
    
    def runner(self, input_):
        """This Function Helps Check if Input is a 
           valid one.
           Reason for this is because, there might be some
           functionalities that are specified to just one 
           image or a video or webcam, this allows to
           efficiently handle that.
           
           @param:
           arg1(input_): the input format to check
        """
        
        try:
            file_ext = int(input_)
        except:
            file_ext =  input_.split('.')[1].lower()

        if file_ext == 0:
            return self.infer_webcam
        elif file_ext in ['png','jpg','jpeg']:
            return self.infer_img
        elif file_ext in ['mp4','mkv']:
            return self.infer_vid
        else:
            raise ValueError(file_ext+' file type not supported try input files with the following extension'' [png, jpg, jpeg, mp4]'' or 0 for webcam')
            
    def check_server(self):
        ### TODO: Connect to the MQTT client ###
        
        #check to see if 
        if self.mqtt_client == None: 
            ValueError('No Server found')
        return 
    
    def get_corr_frame_shape(self,input_shape=None,frame=None):
        """
            This Function is Used to preprocess a single frame to appriopraite
            Size and dimension

            parameters:
                agr1(inputshape) : This is the 4-dim input gotten from an instance of the IENetwork,
                input shoulf be arrange in the format (batch-size, channel, height, width)
                arg2(frame) : This is the image that is about to be passed to the Network for 
                              Inference

            Example:
                get_corr_frame_shape(input_shape=(1,3,416,416),frame=image)
        """
        resized_frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
        resized_frame =  np.expand_dims(resized_frame.transpose((2,0,1))/255,axis=0)
        return resized_frame

    def infer_img(self):
        """
            This Function is Used to pass an image for inference. 

            parameters:
                agr1(inf_net) : This is an Instance of The Network Class, Which has the 
                                architecture and weight loaded in the IENetwork and the
                                Network into the IECore.
                arg2(img_ref) : This is the path to the image on the local machine.

                arg3(out_serveronly): This is a boolean value to indicate if the result 
                                should be saved on the local machine also.
                arg4(out_name): If out_serveronly is False this Value of this augument 
                                will be the file name for the output.

            Example:
                infer_img(inf_net=Network,img_ref="abc/def.jpg",out_serveronly=True,out_name=None)
        """
        ### TODO: Handle the input stream ###
        print('why in img')
        input_shape = self.infer_network.get_input_shape() 
        
        #variable to indicate  the curret input id. the variable can fast forward.
        current_inputid = 0 
        #variable to indicate the current request that we need it get it's request
        curr_needid = 0 
        request_id = 0
        frame_bgr = cv2.imread(self.input_)
        frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        resized_frame = self.get_corr_frame_shape(input_shape, frame_bgr)

        while True:
            key_pressed = cv2.waitKey(60)
            
            #send frame for inference 
            self.frames[current_inputid] = {
                                            'frame':frame_bgr,
                                           }
            if current_inputid == 0:
                self.infer_single_frame(processed_frame=resized_frame,id_=request_id)
                current_inputid += 1
            
            ### Get the output of inference if ready
            if self.check_output_ready(curr_needid):
                #check if to write to disk
                if not self.out_serveronly:
                    # Write out the frame
                    cv2.imwrite(self._out_name,self.frames[curr_needid]['frame'])
                #send output to server
                self.publish_result()
                curr_needid += 1

            # Break if escape key pressed or all output found
            if curr_needid == current_inputid or key_pressed == 27:
                break

        # Release the out writer, capture, and destroy any OpenCV windows
    #     out.release()
    #     cap.release()
        cv2.destroyAllWindows()    
        return

    def infer_vid(self,vid_ref=None,out_serveronly=True,out_name=None):
        """ INferenc on videos """
        ### TODO: Handle the input stream ###
        print('in vid')
        input_shape = self.infer_network.get_input_shape() 
        cap = cv2.VideoCapture(self.input_)
        cap.open(self.input_)
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        # Create a video writer for the output video
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        if self.out_name:
            out = cv2.VideoWriter(self.out_name, 0x00000021, 30, (width,height))
        
        #variable to indicate  the curret input id. the variable can fast forward.
        current_inputid = 0
        #variable to indicate the current request that we need it get it's request
        curr_needid = 0
        request_id = 0
        finalstopflag_ = 0

        while True:
            key_pressed = cv2.waitKey(60)
                
            # Read the next frame
            flag, frame = cap.read()
            print('curr input',current_inputid)
            print('current need', curr_needid)
            if flag:
                resized_frame = self.get_corr_frame_shape(input_shape, frame)
            
                #send frame for inference 
                self.frames[current_inputid] = {
                                                'frame':frame
                                               }
                self.infer_single_frame(processed_frame=resized_frame,id_=request_id)
                current_inputid += 1
            else:
                if (not flag and curr_needid == current_inputid) or key_pressed == 27:
                    # Break if escape key pressed or all output found
                    break
                    
            ### Get the output of inference if ready
            if self.check_output_ready(curr_needid):
                #check if to write to disk
                if not self.out_serveronly:
                    # Write out the frame
                    out.write(self.frames[curr_needid]['frame'])
                #send output to server
                self.publish_result()
                curr_needid += 1

        # Release the out writer, capture, and destroy any OpenCV windows
        if self.out_name:
            out.release()
        cap.release()
        cv2.destroyAllWindows() 

    def infer_webcam(self,out_serveronly=False,out_name=None):
        """ Used to perform Reference on a webcam Image """
        ### TODO: Handle the input stream ###
        print('in cam')
        input_shape = self.infer_network.get_input_shape() 
        cap = cv2.VideoCapture(0)
        cap.open(0)
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        # Create a video writer for the output video
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        if self.out_name:
            out = cv2.VideoWriter(self.out_name, 0x00000021, 30, (width,height))
        
        #variable to indicate  the curret input id. the variable can fast forward.
        current_inputid = 0
        #variable to indicate the current request that we need it get it's request
        curr_needid = 0
        request_id = 0
        finalstopflag_ = 0

        while True:
            key_pressed = cv2.waitKey(1) & 0xFF
                
            # Read the next frame
            flag, frame = cap.read()
            print('curr input',current_inputid)
            print('current need', curr_needid)
            if flag:
                resized_frame = self.get_corr_frame_shape(input_shape, frame)
            
                #send frame for inference 
                self.frames[current_inputid] = {
                                                'frame':frame
                                               }
                self.infer_single_frame(processed_frame=resized_frame,id_=request_id)
                current_inputid += 1
            else:
                if (not flag and curr_needid == current_inputid) or key_pressed == ord("q"):
                    # Break if escape key pressed or all output found
                    break
                    
            ### Get the output of inference if ready
            if self.check_output_ready(curr_needid):
                #check if to write to disk
                if not self.out_serveronly:
                    # Write out the frame
                    out.write(self.frames[curr_needid]['frame'])
                #send output to server
                self.publish_result()
                curr_needid += 1

        # Release the out writer, capture, and destroy any OpenCV windows
        if self.out_name:
            out.release()
        cap.release()
        cv2.destroyAllWindows() 
    
    def infer_single_frame(self,processed_frame=None,id_=None):
        """ Performs inference on a single frame The 3 runners utilize this 
            to run
        """
        if not isinstance(processed_frame,np.ndarray):
            raise ValueError("You don't have an active frame. type ",type(processed_frame)," not a ",np.ndarray," input")

                             
        assert processed_frame.shape == (1,3,416,416)
        assert id_ != None
        if self.async_:
            self.infer_network.exec_net_async(processed_frame, id_=id_)
        else:
            self.infer_network.exec_net_sync(processed_frame)
        return
    
    def get_real_box(self,out,conf_thresh=0.6,batch=1,h=None,w=None,num_anchors=None,only_objectness=1,validation=False,num_classes=80):
        """
        Since Yolo provides 3 different list of output for an image this function helps to remove output that 
        Has it's objectness less than some threshold, also this function help to pick only prediction that has 
        human in it. 
        """
        xs,ys,ws,hs, det_confs,cls_confs = out[0],out[1],out[2],out[3], out[4], out[5:5+num_classes]
        cls_max_confs, cls_max_ids = np.max(cls_confs,axis=0), np.argmax(cls_confs,axis=0)
        cls_confs = cls_confs.transpose(0,1)  
        sz_hw,all_boxes = h*w,[]
        sz_hwa = sz_hw*num_anchors
        for b in range(batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(num_anchors):
                        ind = b*sz_hwa + i*sz_hw + cy*w + cx
                        det_conf =  det_confs[ind]
                        if only_objectness:
                            conf =  det_confs[ind]
                        else:
                            conf = det_confs[ind] * cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        if conf > conf_thresh and cls_max_id in [0,1,2,3,4,5,6,7]:
                            bcx,bcy,bw,bh,cls_max_conf = xs[ind],ys[ind],ws[ind],hs[ind],cls_max_confs[ind]
                            box = [bcx/w, bcy/h, bw/w, bh/w, det_conf, cls_max_conf, cls_max_id]
                            if (not only_objectness) and validation:
                                for c in range(num_classes):
                                    tmp_conf = cls_confs[ind][c]
                                    if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                        box.append(tmp_conf)
                                        box.append(c)
                            boxes.append(box)
            all_boxes.append(boxes)

        return all_boxes


    def print_objects(self,boxes, class_names):    
        print('Objects Found and Confidence Level:\n')
        for i in range(len(boxes)):
            box = boxes[i]
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))


    def plot_boxes(self,img, boxes, class_names, plot_labels, color = None):
        """
        Function to plot boxes on images.
        """
        # Define a tensor used to set the colors of the bounding boxes
        colors = np.array([[255,0,255],[255,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]])

        # Define a function to set the colors of the bounding boxes
        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(np.floor(ratio))
            j = int(np.ceil(ratio))

            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]

            return int(r * 255)

        # Get the width and height of the image
        width = img.shape[1]
        height = img.shape[0]

        # Create a figure and plot the image
        # Plot the bounding boxes and corresponding labels on top of the image
#         for i,box in boxes.items():
        for i in boxes:

            # Get the ith bounding box
            box = boxes[i]['box']
            
            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
            # of the bounding box relative to the size of the image. 
            x1 = int(np.around((box[0] - box[2]/2.0) * width))
            y1 = int(np.around((box[1] - box[3]/2.0) * height))
            x2 = int(np.around((box[0] + box[2]/2.0) * width))
            y2 = int(np.around((box[1] + box[3]/2.0) * height))

            # Set the default rgb value to red
            rgb = (255, 0, 0)

            # Use the same color to plot the bounding boxes of the same object class
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red   = get_color(2, offset, classes) / 255
                green = get_color(1, offset, classes) / 255
                blue  = get_color(0, offset, classes) / 255

                # If a color is given then set rgb to the given color instead
                if color is None:
                    rgb = (red, green, blue)
                else:
                    rgb = color

            # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
            # lower-left corner of the bounding box relative to the size of the image.
            cv2.rectangle(img,(x1, y1),(x2, y2), rgb,2)
            conf_tx = str(class_names[int(cls_id)])+str(i+1)
            cv2.putText(img,conf_tx,(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8,rgb,2)
        return img

    def boxes_iou(self,box1, box2):
        """ Function to calculate how similar two boxes are """
        # Get the Width and Height of each bounding box
        width_box1 = box1[2]
        height_box1 = box1[3]
        width_box2 = box2[2]
        height_box2 = box2[3]

        # Calculate the area of the each bounding box
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2

        # Find the vertical edges of the union of the two bounding boxes
        mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
        Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)

        # Calculate the width of the union of the two bounding boxes
        union_width = Mx - mx

        # Find the horizontal edges of the union of the two bounding boxes
        my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
        My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    

        # Calculate the height of the union of the two bounding boxes
        union_height = My - my

        # Calculate the width and height of the area of intersection of the two bounding boxes
        intersection_width = width_box1 + width_box2 - union_width
        intersection_height = height_box1 + height_box2 - union_height

        # If the the boxes don't overlap then their IOU is zero
        if intersection_width <= 0 or intersection_height <= 0:
            return 0.0

        # Calculate the area of intersection of the two bounding boxes
        intersection_area = intersection_width * intersection_height

        # Calculate the area of the union of the two bounding boxes
        union_area = area_box1 + area_box2 - intersection_area

        # Calculate the IOU
        iou = intersection_area/union_area

        return iou


    def nms(self,boxes, iou_thresh):
        """ This Function takes the output of yolov3 and loops though all
            to know which ones are similar, This function keeps the best 
            prediction for an object
        """
        # If there are no bounding boxes do nothing
        if len(boxes) == 0:
            return boxes

        # Create a PyTorch Tensor to keep track of the detection confidence
        # of each predicted bounding box
        det_confs = np.zeros(len(boxes))

        # Get the detection confidence of each predicted bounding box
        for i in range(len(boxes)):
            det_confs[i] = boxes[i][4]

        # Sort the indices of the bounding boxes by detection confidence value in descending order.
        # We ignore the first returned element since we are only interested in the sorted indices
        # print(f'det_confs --> {det_confs}')
        # _,sortIds = torch.sort(det_confs, descending = True)
        sortIds = cv2.sortIdx(det_confs,-1).reshape(-1)
        # print(f'sort --> {sortIds}')
        # print(f'_ and sortid {_} and {sortIds}')
        # Create an empty list to hold the best bounding boxes after
        # Non-Maximal Suppression (NMS) is performed
        best_boxes = []

        # Perform Non-Maximal Suppression 
        for i in range(len(boxes)):

            # Get the bounding box with the highest detection confidence first
            box_i = boxes[sortIds[i]]

            # Check that the detection confidence is not zero
            if box_i[4] > 0:

                # Save the bounding box 
                best_boxes.append(box_i)

                # Go through the rest of the bounding boxes in the list and calculate their IOU with
                # respect to the previous selected box_i. 
                for j in range(i + 1, len(boxes)):
                    box_j = boxes[sortIds[j]]

                    # If the IOU of box_i and box_j is higher than the given IOU threshold set
                    # box_j's detection confidence to zero. 
                    if self.boxes_iou(box_i, box_j) > iou_thresh:
                        box_j[4] = 0

        return best_boxes

    def detect_objects(self,out=None,iou_thresh=0.3, nms_thresh=0.5):
        """ This Function takes the output of the prediction from the 
            network and sends those output to the get_real_box function
            with the appropraite parameters. The results of the Three 
            outputs are merged into on list and Non-Maximum Suppression
            is performed on them.
        """
        # Start the time. This is done to calculate how long the detection takes.
        def get_pred(out=None,conf_thresh=0.5):
            boxes = list()
            for each in out:
                if each.shape[1] == 507:
                    boxes.append(self.get_real_box(each,conf_thresh=conf_thresh,h=13,w=13,num_anchors=3))
                elif each.shape[1] == 2028:
                    boxes.append(self.get_real_box(each,conf_thresh=conf_thresh,h=26,w=26,num_anchors=3))
                elif each.shape[1] == 8112:
                    boxes.append(self.get_real_box(each,conf_thresh=conf_thresh,h=52,w=52,num_anchors=3))
            return boxes
        #create function to get the real predictions

        out = [each.reshape(-1,85).transpose(1,0) for each in out]   
        list_boxes = get_pred(out,nms_thresh)  

        # Make a new list with all the bounding boxes returned by the neural network
        boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]

        # Perform the second step of NMS on the bounding boxes returned by the neural network.
        # In this step, we only keep the best bounding boxes by eliminating all the bounding boxes
        # whose IOU value is higher than the given IOU threshold
        boxes = self.nms(boxes, iou_thresh)

        # Print the number of objects detected
        print('Number of Objects Detected:', len(boxes))

        return boxes

    def check_output_ready(self,id_=None):
        """ Check if the output is ready to be used, returns 1 if it is, returns 0 if it is not """
        if self.async_:
            "Means it is an async program"
            if self.infer_network.wait(0) == 0:
                result = self.infer_network.get_output(0)
                self.time_taken = time.time() - self.infer_start

            else:
                return 0
        else:
            "means it is a sync program"
            result = self.infer_network.get_output()
            self.time_taken = time.time() - self.infer_start
            
        result = self.detect_objects(result,self.prob_iou,self.prob_threshold)
        result.sort(reverse=1)
        
        ### TODO: Update the frame to include detected bounding boxes
        self.track_boxes(id_,result)
        return 1
    
    def track_boxes(self,id_,result):
        """ This is the human-tracker function, According to my estimation, there are 30 frames in 0.01sec of a video, Given 
            this, each frame was calculated to take 0.00033333333sec to move from one frame to another. Given this, For Each
            Object Detected, We add This constant Value to the total time per frame. The average of the total time for all
            the object detected is what gave the average.
        """
        secs_per_frame = 0.00033333333
        if self.curr_boxes == {} and self.tolerate == {}:
#             self.curr_boxes.update({idx:box for idx,box in zip(range(self.total_detect,self.total_detect+len(result)),result)})
            self.curr_boxes.update({idx:{'box':box, 'time':secs_per_frame} for idx,box in zip(range(self.total_detect,self.total_detect+len(result)),result)})
            self.total_detect += len(self.curr_boxes)
            result.clear()
        else:
            keep = {}          
            # check if current boxes in tolerate
            idx3 = list(self.tolerate.keys())
            idx3.sort()
            _copy_tol = self.tolerate.copy()
            for each_tol_idx in idx3:
                _copy_res = result.copy()
                for each_new in result:
                    iou_curr = self.boxes_iou(each_new,self.tolerate[each_tol_idx]['box'])
                    print('iou res in tolerate boxes  {}'.format(iou_curr))
                    if iou_curr > 0.2:
                        keep.update({each_tol_idx:{'box' : each_new, 'time':secs_per_frame+_copy_tol[each_tol_idx]['time']}})
                        _copy_res.pop(_copy_res.index(each_new))
                        _copy_tol.pop(each_tol_idx)
                        break
                if self.tolerate[each_tol_idx]['count'] == 60 and each_tol_idx not in keep:
                    _copy_tol.pop(each_tol_idx)
                result = _copy_res.copy()
    
            self.tolerate = _copy_tol.copy()
            idx3 = list(self.tolerate.keys())
            idx3.sort()
            
            #increase the chances of boxes in tolorate of not been considered again
            for each_tol_idx in idx3:
                self.tolerate[each_tol_idx]['count'] += 1
                self.tolerate[each_tol_idx]['time'] += secs_per_frame   
                
            # check if current boxes in new boxes else put in tolorate for 30frames before remove   
            idx2 = list(self.curr_boxes.keys())
            idx2.sort()
            _copy_curr = self.curr_boxes.copy()
            for each_old_idx in idx2:
                _copy_res = result.copy()
                for each_new in result:
                    iou_curr = self.boxes_iou(each_new,self.curr_boxes[each_old_idx]['box'])
                    print('iou res {}, boxes {}, {}'.format(iou_curr, each_new, len(self.curr_boxes[each_old_idx])))
                    if iou_curr > 0.2:
                        keep.update({each_old_idx:{'box' : each_new, 'time':secs_per_frame+_copy_curr[each_old_idx]['time']}})
                        _copy_res.pop(_copy_res.index(each_new))
                        _copy_curr.pop(each_old_idx)
                        break
                result = _copy_res.copy()
                
            self.curr_boxes = _copy_curr.copy()
            idx2 = list(self.curr_boxes.keys())
            idx2.sort()
            
            for each_old_idx in idx2:
                self.tolerate.update({each_old_idx:{'box': self.curr_boxes[each_old_idx]['box'], 'time' : secs_per_frame+self.curr_boxes[each_old_idx]['time'], 'count':0}})
            _copy_res = result.copy()
            for each_ in _copy_res:
                keep.update({self.total_detect:{'box':each_, 'time' : secs_per_frame}})
                _copy_res.pop(_copy_res.index(each_))
                self.total_detect += 1
            result = _copy_res.copy()
            self.curr_boxes = keep.copy()
            
        running_avg = 0
        for each in self.curr_boxes:
            running_avg += self.curr_boxes[each]['time']
        running_avg /= len(self.curr_boxes) if len(self.curr_boxes) > 0 else 1

        if self.avg_time == 0 and len(self.curr_boxes) != 0:
            self.avg_time = running_avg
        elif self.avg_time > 0 and len(self.curr_boxes) != 0:
            self.avg_time += running_avg
            self.avg_time /= 2
        print('running average : {}\ntotal average : {}'.format(running_avg, self.avg_time))
        print('curr boxes {} curr total {} curr tolerate {}\n\n'.format(self.curr_boxes,self.total_detect,self.tolerate))
        self.plotted_frame = self.plot_boxes(self.frames[id_]['frame'], self.curr_boxes,self.classes, plot_labels=True)
            
        out_msg = "Inference time: {:.3f}ms"\
                               .format(self.time_taken / 10)
        
        cv2.putText(self.plotted_frame, out_msg, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
             
    def publish_result(self):
        """ This function is called when outputs are to be published """
        ### Send frame to the ffmpeg server
        sys.stdout.buffer.write(self.plotted_frame)
        sys.stdout.flush()   

        #     ### Calculate and send relevant information on ###
        #     ### current_count, total_count and duration to the MQTT server ###
        #     ### Topic "person": keys of "count" and "total" ###
        #     ### Topic "person/duration": key of "duration" ###
        self.mqtt_client.publish("person", json.dumps({"count": len(self.curr_boxes),"total":self.total_detect}))
        self.mqtt_client.publish("person/duration", json.dumps({"duration":round(self.avg_time,4)}))

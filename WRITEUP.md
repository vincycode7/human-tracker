# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## About Model 
The model that was converted for the project is the pytorch's yolov3, It was a hard process but worked out thanks to openvino's team for having enough Supported layers that happened to be in the Yolov3. In the process of converting, Some major Difficulties were encounted, Because the Model had to be converted to Onnx First, There Were some Mis-handling of Some Operators by Onnx as some Operators in the model were treated as though the output of such operators were Constant. After this was discovered, the way i was able to solve that was to use the *.data* on tensors that had to go through Such Operators. I don't know why this solved the error but it did.
In Addition, Since Output Of models were expected to be of fixed size, operations like Non-maximum supression were left for later. The new output of the converted yolov3 is now a list containing 3 elements of shapes [n*13*13*255, n*26*26*255, n*52*52*255], where n is batch size. this output allows the model to be correctly converted to onnx and then to IR representation. SO During Inference, The output of the IR model will be processed by Perform NMS on the output to get correct predictions.

## Link To the Pytorch model
https://modelzoo.co/model/pytorch-yolov3-pytorch

## Link to .bin file for IR model
Since I can push large files to Github, the weight(.bin) of the IR model is saved in drive and can be downloaded into the model folder.

[To Download weight]
'''
cd model
'''
Goto https://drive.google.com/drive/folders/164aCzkNrhvndhAJaMrUbjnTB4PXLQlac
download the weight and move it to the model folder


## To Run Inference on the model
reset && python3 main.py -d CPU -i resources/Pedestrian_Detect_2_1_1.mp4 -l "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so" -m models/project7.xml -pt 0.5 -iou 0.2 -md 1 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

##[Note] :- Any other model can't be used to test this software as input to this model is expected to be 1*416*416, code to take care of this is in the software though.
So to successfully run this model the weight has to be downloaded.

## Explaining Custom Layers

[Answer]
So, Custom layers are layers that make up our model but are not supported by openvino. For Instance If anywhere in the model a Layer is introduced that is not recognized by Openvino, Then this will be tagged as a Custom Layers. So to Use them in our model We have to register the Custom Layers So that openvino can successfully pass inputs through them and get the correct output.

The process behind converting custom layers involves...

[Answer]
                           [For Tensorflow]
(0) In TensorFlow, the first option is to register the custom layers as extensions to the Model Optimizer.
(1) Its second option is to actually replace the unsupported subgraph with a different subgraph.
(2) The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

                           [For Caffe]
(0) In Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.
(1) The second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.

Some of the potential reasons for handling custom layers are...
[Answer]
(0) So That the input to the layer can be correctly calculated
(1) So the Output can be corectly calculated

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

[Answer]
(0) Accuracy of model
(1) Size Of Model
(2) Inference time of model

The difference between model accuracy pre- and post-conversion was...

[Answer]
There was no observable change in accuracy When convertion was completed

The size of the model pre- and post-conversion was...

[Answer]
The model size is 247mb before convertion but when i converted to FP16 i got half the size at 127mb and i got the same size for FP32 at 247mb.

The inference time of the model pre- and post-conversion was...

[Answer]
For Inference time i most confess, I Received a very Big Boost from convertion. Before Training i Got an Average of 1.8seconds on inference per image but after convertion i got 0.03secs per image.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

[Answer]
(0) First Of, i am working on a facial Verification model that helps compare two faces if they match, i will be Using the App Soon to implement that for every human tracked.
(1) Also this Can Be Used To Detect Congested Areas That might need the Supervision Of the Authority.

Each of these use cases would be useful because...

## Assess Effects on End User Needs
(0) Facial Verfication can be Used to Check for Criminals, Find Lots Children or Family Members, and if human tracking and facial Verification can be implemented together, that can provide some level of security to the people.
(1) Congested Areas Can Slow Business and Make People Run Late at work, we don't want people been fired, SO this kind of application if very Useful.

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
(0) As Lighting can Cause some kind of variation in the input image that model might not be able to detect some object due to this, SO Lighting can make model make few mistakes in prediction.
(1) Model accuracy is very important as Medical Related models should be accurate enough to prevent the death of a person, So if a model is to be deployed at all, it should be accurate enough to prevent serious damages.
(2) Most models have Fixed Sizes as their Inputs, but because image size of cameras vary, one might have to do a bit of preprocessing which involes resizing the image that might also result in making objects too small or too big to be detected. This can affect accuracy.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

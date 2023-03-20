## Running inference with YOLOV5, Onnx and OpenCV on C++
# Exporting the YOLOV5 model as .onnx
If you want to train and export your own custom onnx model you can follow the steps that are set up in the following Google Colab: https://colab.research.google.com/drive/19kVzBERhRwB1jywcKeJ3dALARNd5-dR7?usp=sharing

After you have succesfully exported your custom trained model as a .onnx file you can create a .txt file containing the names of your classes as seen in the example file. I'ld advice to leave an enter space between each class name.

Now you can modify the path/name of your custom files in the main.cpp file and run your inference!

**Note**: Remember to check out the original code that was modified in this project at Doleron's github https://github.com/doleron/yolov5-opencv-cpp-python

# Resnet-18 using mediapipe

This is a simple mediapipe project that uses a resnet-18 model pretrained on imagenet in a mediapipe pipeline. The pretrained model was converted from a pytorch model to tflite, and then included in the pipeline.<br/>The pytorch model was converted to tflite using this [gist](https://gist.github.com/firekind/c98ae11f607b22ebf0ea832ebd88f3a1).

## Build
to build the project, run
```
$ bazel build -c opt --copt -DEGL_NO_X11 resnet-mediapipe:resnet-mediapipe
```

and to run the project, run
```
$ ./bazel-bin/resnet-mediapipe/resnet-mediapipe --calculator_graph_config_file=./resnet-mediapipe/graphs/resnet.pbtxt --image_path=./images/input.jpg
```
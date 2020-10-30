load("@mediapipe//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_simple_subgraph")

package(default_visibility = ["//visibility:public"])


mediapipe_simple_subgraph(
    name = "resnet_subgraph",
    graph = "resnet_subgraph.pbtxt",
    register_as = "ResNetSubgraph",
    deps = [
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_converter_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_inference_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_classification_calculator",
        "//resnet-mediapipe:hwc_to_chw_calculator",
    ],
)

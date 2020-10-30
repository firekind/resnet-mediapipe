load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "hwc_to_chw_calculator",
    srcs = ["src/hwc_to_chw_calculator.cc"],
    hdrs = ["include/hwc_to_chw_calculator.h"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True
)

cc_binary(
    name = "resnet-mediapipe",
    srcs = ["main.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/formats:classification_cc_proto",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:commandlineflags",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:logging",
        "//resnet-mediapipe/graphs:resnet_calculators",
    ],
)

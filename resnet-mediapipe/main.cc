#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"

constexpr char kInputStream[] = "in";
constexpr char kOutputStream[] = "out";

DEFINE_string(
    calculator_graph_config_file, 
    "",
    "Name of the file containing the graph"
);

DEFINE_string(
    image_path,
    "",
    "Path to the input image"
);

namespace mediapipe {
    Status run_mpp_graph() {
        // getting graph file contents
        std::string graph_file_contents;
        MP_RETURN_IF_ERROR(
            file::GetContents(
                FLAGS_calculator_graph_config_file,
                &graph_file_contents
            )
        );

        LOG(INFO) << "Parsing graph file" << graph_file_contents;
        CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(
            graph_file_contents
        );

        LOG(INFO) << "Initializing graph";
        CalculatorGraph graph;
        MP_RETURN_IF_ERROR(graph.Initialize(config));

        LOG(INFO) << "Loading image";
        // loading image using opencv
        cv::Mat image_mat = cv::imread(FLAGS_image_path, cv::IMREAD_COLOR);
        cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2RGB);
        // converting image to image frame
        auto image_frame = absl::make_unique<ImageFrame>(
            ImageFormat::SRGB,
            image_mat.cols,
            image_mat.rows,
            ImageFrame::kDefaultAlignmentBoundary
        );
        cv::Mat image_frame_mat = formats::MatView(image_frame.get());
        image_mat.copyTo(image_frame_mat);

        LOG(INFO) << "Starting graph";
        ASSIGN_OR_RETURN(
            OutputStreamPoller poller,
            graph.AddOutputStreamPoller(kOutputStream)
        );
        MP_RETURN_IF_ERROR(graph.StartRun({}));

        LOG(INFO) << "Passing image into graph";
        MP_RETURN_IF_ERROR(
            graph.AddPacketToInputStream(
                kInputStream,
                Adopt(image_frame.release()).At(Timestamp().NextAllowedInStream())
            )
        );

        LOG(INFO) << "Getting result from graph";
        Packet output_packet;
        RET_CHECK(poller.Next(&output_packet));
        
        auto& output = output_packet.Get<ClassificationList>();
        std::cout << output.classification(0).label();
        

        LOG(INFO) << "Shutting down";
        MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
        return graph.WaitUntilDone();
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ::mediapipe::Status run_status = mediapipe::run_mpp_graph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}

// ------------------------------------------------------------
// Performs object detection using YOLOv8 (ONNX format) with OpenCV DNN.
// 1. Loads a YOLOv8 model
// 2. Preprocesses an input image
// 3. Runs inference on the image
// 4. Applies Non-Maximum Suppression (NMS)
// 5. Draws bounding boxes and class labels
// ------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Struct to hold one detection result

struct Detection {
    Rect box;         // Bounding box (x, y, width, height)
    float confidence; // Confidence score (0–1)
    int classId;      // Class index (integer class label)
};


// Function: Non-Maximum Suppression (NMS)
// Removes overlapping bounding boxes based on IoU threshold.
// Keeps only the most confident detections per object.

void nms(vector<Detection>& detections, float nmsThreshold) {
    if (detections.empty()) return;

    // Sort detections by descending confidence (best first)
    sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    vector<bool> suppressed(detections.size(), false);

    // For each detection, compare with the rest
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            // Compute IoU (Intersection over Union)
            Rect inter = detections[i].box & detections[j].box;
            Rect uni = detections[i].box | detections[j].box;
            float iou = (uni.area() > 0) ? (float)inter.area() / (float)uni.area() : 0.0f;

            // Suppress box if IoU is above threshold
            if (iou > nmsThreshold)
                suppressed[j] = true;
        }
    }

    // Keep only non-suppressed detections
    vector<Detection> finalDetections;
    for (size_t i = 0; i < detections.size(); ++i)
        if (!suppressed[i]) finalDetections.push_back(detections[i]);

    detections = move(finalDetections);
}


// Function: loadClassNames
// Reads COCO class names (e.g., "person", "car") from a text file

vector<string> loadClassNames(const string& filename) {
    vector<string> class_names;
    ifstream ifs(filename);
    string line;
    while (getline(ifs, line)) {
        if (!line.empty()) class_names.push_back(line);
    }
    return class_names;
}


// Main program entry point

int main() {
    // File paths 
    string model_path = "yolov8n.onnx";   // YOLOv8 model in ONNX format
    string class_file = "coco.names";     // List of class names (COCO dataset)
    string image_path = "image1.jpg";     // Input image

    // Load class names
    vector<string> class_names = loadClassNames(class_file);
    cout << " Loaded classes: " << class_names.size() << endl;

    // Load and validate the input image
    Mat image = imread(image_path);
    if (image.empty()) {
        cerr << " Could not open image: " << image_path << endl;
        return -1;
    }
    int origW = image.cols, origH = image.rows; // Save original size

    
    // YOLOv8 model input size (default: 640x640)
    const int inputSize = 640;

    // Resize image to match YOLO input (no padding / letterbox)
    Mat resized;
    resize(image, resized, Size(inputSize, inputSize));

    // Create a 4D blob from image for network input
    // - Scale factor: 1/255 (normalize pixel values)
    // - swapRB = true (BGR → RGB)
    Mat blob = blobFromImage(resized, 1.0 / 255.0, Size(inputSize, inputSize),
                             Scalar(), true, false, CV_32F);

    // Load YOLOv8 model and set computation backend/target
    Net net = readNet(model_path);
    net.setPreferableBackend(DNN_BACKEND_OPENCV); // use OpenCV backend
    net.setPreferableTarget(DNN_TARGET_CPU);      // run on CPU (can use CUDA if built)

    // Perform forward inference
    net.setInput(blob); // Set preprocessed image as input
    vector<Mat> outputs;

    auto t0 = chrono::high_resolution_clock::now();
    net.forward(outputs, net.getUnconnectedOutLayersNames()); // Run network
    auto t1 = chrono::high_resolution_clock::now();

    double infTime = chrono::duration<double>(t1 - t0).count();
    cout << " Inference Time: " << fixed << setprecision(4) << infTime << " s" << endl;

    // Check if inference produced output
    if (outputs.empty()) {
        cerr << " No outputs from network." << endl;
        return -1;
    }

    // Inspect output tensor shape
    Mat out = outputs[0];
    cout << " Output dims: " << out.dims << endl;
    for (int i = 0; i < out.dims; ++i)
        cout << " - size[" << i << "] = " << out.size[i] << endl;

    // Vector to store all detection results
    vector<Detection> detections;

    // Scaling factors to map model output → original image size
    float scaleX = (float)origW / (float)inputSize;
    float scaleY = (float)origH / (float)inputSize;

    // Decode YOLOv8 output tensor
    // Typical format: [1, 84, 8400] or [1, 85, 8400]
    // [cx, cy, w, h, (obj_conf), class_scores...]
    if (out.dims == 3) {
        int ch = out.size[1];       // Number of channels per detection (e.g., 85)
        int detCount = out.size[2]; // Number of detections (e.g., 8400)

        // Ensure output shape matches expected YOLOv8 format
        if ((ch == 84 || ch == 85) && detCount == 8400) {

            // Reshape tensor from [1, ch, N] → [N, ch]
            Mat reshaped = out.reshape(1, ch).t();
            bool has_objectness = (ch == 85); // YOLOv8 includes objectness confidence

            // Loop through each detection
            for (int i = 0; i < reshaped.rows; ++i) {
                float* data = reshaped.ptr<float>(i);

                // Extract bounding box center and size (normalized to [0,1])
                float cx = data[0] * inputSize;
                float cy = data[1] * inputSize;
                float w  = data[2] * inputSize;
                float h  = data[3] * inputSize;

                // Extract confidence scores
                float objectness = has_objectness ? data[4] : 1.0f;
                int class_offset = has_objectness ? 5 : 4;
                int num_classes = ch - class_offset;
                if (num_classes <= 0) continue;

                // Find class with highest score
                Mat scores(1, num_classes, CV_32F, data + class_offset);
                Point classIdPoint;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

                // Combine objectness and class confidence
                float conf = objectness * (float)maxClassScore;
                if (conf <= 0.3f) continue; // skip low-confidence boxes

                int cls = classIdPoint.x;
                if (cls < 0 || cls >= (int)class_names.size()) continue;

                // Convert (cx, cy, w, h) → (x, y, width, height)
                float x1 = cx - w / 2.0f;
                float y1 = cy - h / 2.0f;

                // Scale box to original image size
                int rx = int(round(x1 * scaleX));
                int ry = int(round(y1 * scaleY));
                int rw = int(round(w * scaleX));
                int rh = int(round(h * scaleY));

                // Clip box to image boundaries
                rx = max(0, min(rx, origW - 1));
                ry = max(0, min(ry, origH - 1));
                if (rw <= 0 || rh <= 0) continue;
                if (rx + rw > origW) rw = origW - rx;
                if (ry + rh > origH) rh = origH - ry;

                // Store valid detection
                Detection d{Rect(rx, ry, rw, rh), conf, cls};
                detections.push_back(d);
            }
        } else {
            cerr << " Unexpected output shape: [" << out.size[0] << "," 
                 << out.size[1] << "," << out.size[2] << "]\n";
        }
    } else {
        cerr << " Unexpected output dims: " << out.dims << endl;
    }

    cout << " Raw detections before NMS: " << detections.size() << endl;

    // Apply Non-Maximum Suppression to reduce overlaps
    float nmsThreshold = 0.45f;
    nms(detections, nmsThreshold);
    cout << " Detections after NMS: " << detections.size() << endl;

    // Draw detected boxes and labels on the image
    for (const auto &d : detections) {
        // Draw bounding box
        rectangle(image, d.box, Scalar(0, 255, 0), 2);

        // Draw label text above the box
        stringstream ss;
        ss << class_names[d.classId] << " " 
           << fixed << setprecision(2) << d.confidence;
        putText(image, ss.str(), Point(d.box.x, max(0, d.box.y - 5)),
                FONT_HERSHEY_SIMPLEX, 2, Scalar(0,255,0), 3);
    }

    // Resize image for display window
    int displayWidth = 800;
    float scale = (float)displayWidth / image.cols;
    int displayHeight = int(image.rows * scale);
    Mat display;
    resize(image, display, Size(displayWidth, displayHeight));

    // Display and save results
    imshow("YOLOv8 Detections", display);
    imwrite("output_result.jpg", image); // save full-resolution annotated image

    waitKey(0); // Wait for key press before closing window
    return 0;
}

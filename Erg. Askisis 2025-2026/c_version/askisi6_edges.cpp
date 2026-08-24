#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static fs::path find_file(const fs::path &root, const std::string &filename)
{
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().filename().string() == filename) {
            return entry.path();
        }
    }

    throw std::runtime_error("Could not find input file: " + filename);
}

static cv::Mat imread_grayscale_any_path(const fs::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open image: " + path.string());
    }

    std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
    cv::Mat decoded = cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
    if (decoded.empty()) {
        throw std::runtime_error("Could not decode image: " + path.string());
    }
    return decoded;
}

static cv::Mat normalize_to_uint8(const cv::Mat &image)
{
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(image, &min_value, &max_value);

    if (max_value == min_value) {
        return cv::Mat::zeros(image.size(), CV_8U);
    }

    cv::Mat normalized;
    image.convertTo(normalized, CV_8U, 255.0 / (max_value - min_value), -255.0 * min_value / (max_value - min_value));
    return normalized;
}

static cv::Mat panel(const cv::Mat &image, const std::string &title, int target_width = 360)
{
    cv::Mat color;
    if (image.channels() == 1) {
        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    } else {
        color = image.clone();
    }

    const double scale = static_cast<double>(target_width) / color.cols;
    cv::Mat resized;
    cv::resize(color, resized, cv::Size(target_width, std::max(1, static_cast<int>(color.rows * scale))));

    const int title_height = 40;
    cv::Mat canvas(resized.rows + title_height, resized.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    resized.copyTo(canvas(cv::Rect(0, title_height, resized.cols, resized.rows)));
    cv::putText(canvas, title, cv::Point(10, 27), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    return canvas;
}

static cv::Mat pad_to_height(const cv::Mat &image, int target_height)
{
    if (image.rows >= target_height) {
        return image;
    }

    cv::Mat padded(target_height, image.cols, image.type(), cv::Scalar(255, 255, 255));
    image.copyTo(padded(cv::Rect(0, 0, image.cols, image.rows)));
    return padded;
}

static void save_row_panel(std::vector<cv::Mat> panels, const fs::path &path)
{
    int max_height = 0;
    for (const cv::Mat &item : panels) {
        max_height = std::max(max_height, item.rows);
    }
    for (cv::Mat &item : panels) {
        item = pad_to_height(item, max_height);
    }

    cv::Mat combined;
    cv::hconcat(panels, combined);
    cv::imwrite(path.string(), combined);
}

int main()
{
    try {
        const fs::path root = fs::path("..");
        const fs::path image_path = find_file(root / "Images", "hallway.png");
        const fs::path output_dir = root / "outputs" / "askisi6_edges_cpp";
        fs::create_directories(output_dir);

        const cv::Mat original = imread_grayscale_any_path(image_path);

        cv::Mat gx;
        cv::Mat gy;
        cv::Sobel(original, gx, CV_64F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REFLECT);
        cv::Sobel(original, gy, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT);

        cv::Mat abs_gx = normalize_to_uint8(cv::abs(gx));
        cv::Mat abs_gy = normalize_to_uint8(cv::abs(gy));

        cv::Mat magnitude;
        cv::magnitude(gx, gy, magnitude);
        cv::Mat magnitude_uint8 = normalize_to_uint8(magnitude);

        cv::Mat edge_map;
        const double threshold_value = cv::threshold(magnitude_uint8, edge_map, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edge_map, lines, 1.0, CV_PI / 180.0, 10, 80.0, 10.0);

        cv::Mat overlay;
        cv::cvtColor(original, overlay, cv::COLOR_GRAY2BGR);
        for (const cv::Vec4i &line : lines) {
            cv::line(overlay, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }

        cv::imwrite((output_dir / "sobel_x.png").string(), abs_gx);
        cv::imwrite((output_dir / "sobel_y.png").string(), abs_gy);
        cv::imwrite((output_dir / "gradient_magnitude.png").string(), magnitude_uint8);
        cv::imwrite((output_dir / "edge_map.png").string(), edge_map);
        cv::imwrite((output_dir / "lines_overlay.png").string(), overlay);

        save_row_panel(
            {
                panel(original, "Original"),
                panel(abs_gx, "|Sobel X|"),
                panel(abs_gy, "|Sobel Y|")
            },
            output_dir / "sobel_summary.png"
        );

        std::ostringstream threshold_title;
        threshold_title << "Thresholded Edges (T=" << threshold_value << ")";
        save_row_panel(
            {
                panel(magnitude_uint8, "Gradient Magnitude"),
                panel(edge_map, threshold_title.str()),
                panel(overlay, "Detected Lines Overlay")
            },
            output_dir / "edges_and_lines_summary.png"
        );

        std::ostringstream summary;
        summary << "{\n";
        summary << "  \"threshold_value\": " << threshold_value << ",\n";
        summary << "  \"detected_line_segments\": " << lines.size() << ",\n";
        summary << "  \"line_color_rgb\": [255, 0, 0],\n";
        summary << "  \"line_width\": 3\n";
        summary << "}\n";

        std::ofstream summary_file(output_dir / "summary.json");
        summary_file << summary.str();

        std::cout << summary.str();
        std::cout << "Outputs saved to: " << output_dir.string() << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}

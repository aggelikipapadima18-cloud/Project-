#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const cv::Size ADAPTIVE_KERNEL_SIZE(64, 64);
static const cv::Size ADAPTIVE_CANDIDATES[] = {cv::Size(32, 32), cv::Size(64, 64), cv::Size(128, 128)};

static std::vector<fs::path> find_dark_road_images(const fs::path &root)
{
    std::vector<fs::path> paths;
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("dark_road_", 0) == 0 && entry.path().extension().string() == ".jpg") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
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

static cv::Mat clahe_with_kernel(const cv::Mat &image, cv::Size kernel_size)
{
    const int tiles_x = std::max(1, image.cols / kernel_size.width);
    const int tiles_y = std::max(1, image.rows / kernel_size.height);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(tiles_x, tiles_y));
    cv::Mat result;
    clahe->apply(image, result);
    return result;
}

static double stddev_gray(const cv::Mat &image)
{
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(image, mean, stddev);
    return stddev[0];
}

static double noise_proxy(const cv::Mat &image)
{
    cv::Mat as_double;
    image.convertTo(as_double, CV_64F);

    cv::Mat gx = cv::Mat::zeros(image.size(), CV_64F);
    cv::Mat gy = cv::Mat::zeros(image.size(), CV_64F);

    for (int y = 0; y < image.rows; ++y) {
        const double *src = as_double.ptr<double>(y);
        double *dx = gx.ptr<double>(y);
        for (int x = 0; x < image.cols; ++x) {
            dx[x] = src[x] - src[std::max(0, x - 1)];
        }
    }

    for (int y = 0; y < image.rows; ++y) {
        const double *src = as_double.ptr<double>(y);
        const double *prev = as_double.ptr<double>(std::max(0, y - 1));
        double *dy = gy.ptr<double>(y);
        for (int x = 0; x < image.cols; ++x) {
            dy[x] = src[x] - prev[x];
        }
    }

    cv::Scalar mean_x;
    cv::Scalar std_x;
    cv::Scalar mean_y;
    cv::Scalar std_y;
    cv::meanStdDev(gx, mean_x, std_x);
    cv::meanStdDev(gy, mean_y, std_y);
    return std_x[0] + std_y[0];
}

static cv::Mat draw_histogram(const cv::Mat &image)
{
    const int width = 420;
    const int height = 240;
    const int margin = 24;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    int hist_size = 256;
    float range[] = {0.0f, 256.0f};
    const float *hist_range[] = {range};
    cv::Mat hist;
    cv::calcHist(&image, 1, nullptr, cv::Mat(), hist, 1, &hist_size, hist_range);

    double max_value = 0.0;
    cv::minMaxLoc(hist, nullptr, &max_value);
    max_value = std::max(max_value, 1.0);

    cv::line(canvas, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 1);
    cv::line(canvas, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(0, 0, 0), 1);

    for (int i = 1; i < hist_size; ++i) {
        const int x0 = margin + (i - 1) * (width - 2 * margin) / (hist_size - 1);
        const int x1 = margin + i * (width - 2 * margin) / (hist_size - 1);
        const int y0 = height - margin - static_cast<int>(hist.at<float>(i - 1) * (height - 2 * margin) / max_value);
        const int y1 = height - margin - static_cast<int>(hist.at<float>(i) * (height - 2 * margin) / max_value);
        cv::line(canvas, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 0, 0), 1);
    }

    return canvas;
}

static cv::Mat panel(const cv::Mat &image, const std::string &title, int target_width = 420)
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
    cv::putText(canvas, title, cv::Point(10, 27), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
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

static cv::Mat hconcat_same_height(std::vector<cv::Mat> images)
{
    int max_height = 0;
    for (const cv::Mat &image : images) {
        max_height = std::max(max_height, image.rows);
    }

    for (cv::Mat &image : images) {
        image = pad_to_height(image, max_height);
    }

    cv::Mat combined;
    cv::hconcat(images, combined);
    return combined;
}

static void save_histogram_figure(const cv::Mat &original, const cv::Mat &equalized, const cv::Mat &adaptive, const fs::path &path)
{
    std::vector<cv::Mat> rows;
    const std::vector<std::pair<std::string, cv::Mat>> entries = {
        {"Original", original},
        {"Global HE", equalized},
        {"Adaptive HE 64x64", adaptive}
    };

    for (const auto &entry : entries) {
        cv::Mat left = panel(entry.second, entry.first);
        cv::Mat right = panel(draw_histogram(entry.second), entry.first + " Histogram");
        rows.push_back(hconcat_same_height({left, right}));
    }

    cv::Mat combined;
    cv::vconcat(rows, combined);
    cv::imwrite(path.string(), combined);
}

static void save_adaptive_candidates_figure(const cv::Mat &original, const std::vector<std::pair<cv::Size, cv::Mat>> &variants, const fs::path &path)
{
    std::vector<cv::Mat> panels;
    panels.push_back(panel(original, "Original", 300));

    for (const auto &variant : variants) {
        std::ostringstream title;
        title << "CLAHE " << variant.first.width << "x" << variant.first.height;
        panels.push_back(panel(variant.second, title.str(), 300));
    }

    cv::Mat combined = hconcat_same_height(panels);
    cv::imwrite(path.string(), combined);
}

static std::string process_image(const fs::path &path, const fs::path &output_dir)
{
    const cv::Mat image = imread_grayscale_any_path(path);

    cv::Mat global_he;
    cv::equalizeHist(image, global_he);

    const cv::Mat adaptive_he = clahe_with_kernel(image, ADAPTIVE_KERNEL_SIZE);

    std::vector<std::pair<cv::Size, cv::Mat>> candidate_images;
    std::ostringstream candidate_lines;
    for (const cv::Size &kernel_size : ADAPTIVE_CANDIDATES) {
        cv::Mat adaptive_candidate = clahe_with_kernel(image, kernel_size);
        candidate_images.push_back({kernel_size, adaptive_candidate});
        candidate_lines
            << "kernel=(" << kernel_size.width << ", " << kernel_size.height << ") "
            << "contrast_std=" << stddev_gray(adaptive_candidate) << " "
            << "noise_proxy=" << noise_proxy(adaptive_candidate) << "\n";
    }

    const std::string base_name = path.stem().string();
    cv::imwrite((output_dir / (base_name + "_global_he.png")).string(), global_he);
    cv::imwrite((output_dir / (base_name + "_adaptive_he.png")).string(), adaptive_he);
    save_histogram_figure(image, global_he, adaptive_he, output_dir / (base_name + "_histograms.png"));
    save_adaptive_candidates_figure(image, candidate_images, output_dir / (base_name + "_adaptive_candidates.png"));

    std::ofstream metrics_file(output_dir / (base_name + "_adaptive_metrics.txt"));
    metrics_file << candidate_lines.str();

    std::ostringstream line;
    line
        << base_name
        << ": original_std=" << stddev_gray(image)
        << ", global_std=" << stddev_gray(global_he)
        << ", adaptive_std=" << stddev_gray(adaptive_he)
        << ", adaptive_noise_proxy=" << noise_proxy(adaptive_he);
    return line.str();
}

int main()
{
    try {
        const fs::path root = fs::path("..");
        const fs::path output_dir = root / "outputs" / "askisi2_histogram_equalization_cpp";
        fs::create_directories(output_dir);

        const std::vector<fs::path> image_paths = find_dark_road_images(root / "Images");
        if (image_paths.empty()) {
            throw std::runtime_error("No dark_road_*.jpg images were found.");
        }

        std::ostringstream summary;
        summary << "Adaptive kernel size selected: (64, 64)\n";
        summary << "Candidate kernels: (32, 32), (64, 64), (128, 128)\n\n";

        for (const fs::path &image_path : image_paths) {
            summary << process_image(image_path, output_dir) << "\n";
        }

        std::ofstream summary_file(output_dir / "summary.txt");
        summary_file << summary.str();

        std::cout << summary.str();
        std::cout << "Outputs saved to: " << output_dir.string() << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}

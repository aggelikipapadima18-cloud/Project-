#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static const int BLOCK_SIZE = 32;
static const int R_VALUES[] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
static const int SHOWCASE_R_VALUES[] = {10, 25, 50};

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

    cv::Mat result;
    decoded.convertTo(result, CV_64F);
    return result;
}

static cv::Mat clip_to_uint8(const cv::Mat &image)
{
    cv::Mat bounded;
    cv::Mat clipped;
    cv::max(image, 0.0, bounded);
    cv::min(bounded, 255.0, bounded);
    bounded.convertTo(clipped, CV_8U);
    return clipped;
}

static cv::Mat pad_to_block_size(const cv::Mat &image, int block_size)
{
    const int padded_rows = static_cast<int>(std::ceil(static_cast<double>(image.rows) / block_size) * block_size);
    const int padded_cols = static_cast<int>(std::ceil(static_cast<double>(image.cols) / block_size) * block_size);
    const int pad_bottom = padded_rows - image.rows;
    const int pad_right = padded_cols - image.cols;

    cv::Mat padded;
    cv::copyMakeBorder(image, padded, 0, pad_bottom, 0, pad_right, cv::BORDER_REPLICATE);
    return padded;
}

static std::vector<cv::Point> zigzag_indices(int size)
{
    std::vector<cv::Point> order;
    for (int diag = 0; diag < 2 * size - 1; ++diag) {
        std::vector<cv::Point> coords;
        const int row_start = std::max(0, diag - size + 1);
        const int row_end = std::min(diag, size - 1);
        for (int row = row_start; row <= row_end; ++row) {
            const int col = diag - row;
            coords.push_back(cv::Point(col, row));
        }
        if (diag % 2 == 0) {
            std::reverse(coords.begin(), coords.end());
        }
        order.insert(order.end(), coords.begin(), coords.end());
    }
    return order;
}

static cv::Mat zonal_mask(int size, int keep_count)
{
    cv::Mat mask = cv::Mat::zeros(size, size, CV_64F);
    const std::vector<cv::Point> order = zigzag_indices(size);
    for (int i = 0; i < keep_count && i < static_cast<int>(order.size()); ++i) {
        mask.at<double>(order[i].y, order[i].x) = 1.0;
    }
    return mask;
}

static cv::Mat dct_blocks(const cv::Mat &padded)
{
    cv::Mat result = cv::Mat::zeros(padded.size(), CV_64F);
    for (int y = 0; y < padded.rows; y += BLOCK_SIZE) {
        for (int x = 0; x < padded.cols; x += BLOCK_SIZE) {
            cv::Mat block = padded(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE));
            cv::Mat transformed;
            cv::dct(block, transformed);
            transformed.copyTo(result(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)));
        }
    }
    return result;
}

static cv::Mat inverse_dct_blocks(const cv::Mat &coefficients)
{
    cv::Mat result = cv::Mat::zeros(coefficients.size(), CV_64F);
    for (int y = 0; y < coefficients.rows; y += BLOCK_SIZE) {
        for (int x = 0; x < coefficients.cols; x += BLOCK_SIZE) {
            cv::Mat block = coefficients(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE));
            cv::Mat restored;
            cv::idct(block, restored);
            restored.copyTo(result(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)));
        }
    }
    return result;
}

static cv::Mat apply_zonal(const cv::Mat &coefficients, int keep_count)
{
    cv::Mat result = cv::Mat::zeros(coefficients.size(), CV_64F);
    const cv::Mat mask = zonal_mask(BLOCK_SIZE, keep_count);

    for (int y = 0; y < coefficients.rows; y += BLOCK_SIZE) {
        for (int x = 0; x < coefficients.cols; x += BLOCK_SIZE) {
            cv::Mat source = coefficients(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE));
            cv::Mat destination = result(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE));
            cv::multiply(source, mask, destination);
        }
    }

    return result;
}

static cv::Mat apply_threshold(const cv::Mat &coefficients, int keep_count)
{
    cv::Mat result = cv::Mat::zeros(coefficients.size(), CV_64F);

    for (int y = 0; y < coefficients.rows; y += BLOCK_SIZE) {
        for (int x = 0; x < coefficients.cols; x += BLOCK_SIZE) {
            cv::Mat source = coefficients(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE));
            std::vector<std::pair<double, cv::Point>> ranked;
            ranked.reserve(BLOCK_SIZE * BLOCK_SIZE);

            for (int row = 0; row < BLOCK_SIZE; ++row) {
                for (int col = 0; col < BLOCK_SIZE; ++col) {
                    ranked.push_back({std::abs(source.at<double>(row, col)), cv::Point(col, row)});
                }
            }

            std::nth_element(
                ranked.begin(),
                ranked.begin() + keep_count,
                ranked.end(),
                [](const auto &left, const auto &right) { return left.first > right.first; }
            );

            cv::Mat destination = result(cv::Rect(x, y, BLOCK_SIZE, BLOCK_SIZE));
            for (int i = 0; i < keep_count; ++i) {
                const cv::Point point = ranked[i].second;
                destination.at<double>(point.y, point.x) = source.at<double>(point.y, point.x);
            }
        }
    }

    return result;
}

static double mse(const cv::Mat &reference, const cv::Mat &test)
{
    cv::Mat diff;
    cv::subtract(reference, test, diff);
    cv::multiply(diff, diff, diff);
    return cv::mean(diff)[0];
}

static cv::Mat crop_original_size(const cv::Mat &image, cv::Size size)
{
    return image(cv::Rect(0, 0, size.width, size.height)).clone();
}

static cv::Mat panel(const cv::Mat &image, const std::string &title, int target_width = 340)
{
    cv::Mat color;
    cv::cvtColor(clip_to_uint8(image), color, cv::COLOR_GRAY2BGR);

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

static void save_comparison_figure(const cv::Mat &original, const std::vector<std::pair<int, cv::Mat>> &reconstructions, const std::string &method_name, const fs::path &path)
{
    std::vector<cv::Mat> panels;
    panels.push_back(panel(original, "Original"));
    for (const auto &item : reconstructions) {
        std::ostringstream title;
        title << method_name << " r=" << item.first << "%";
        panels.push_back(panel(item.second, title.str()));
    }

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

static void save_mse_curve(const std::vector<int> &r_values, const std::vector<double> &zonal_mse, const std::vector<double> &threshold_mse, const fs::path &path)
{
    const int width = 820;
    const int height = 500;
    const int left = 80;
    const int right = 30;
    const int top = 45;
    const int bottom = 70;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    std::vector<double> all_errors = zonal_mse;
    all_errors.insert(all_errors.end(), threshold_mse.begin(), threshold_mse.end());
    const double min_y = *std::min_element(all_errors.begin(), all_errors.end());
    const double max_y = *std::max_element(all_errors.begin(), all_errors.end());
    const double y_range = std::max(max_y - min_y, 1e-12);
    const double min_x = r_values.front();
    const double max_x = r_values.back();

    auto to_point = [&](int r, double error) {
        const double px = left + (r - min_x) * (width - left - right) / std::max(max_x - min_x, 1.0);
        const double py = height - bottom - (error - min_y) * (height - top - bottom) / y_range;
        return cv::Point(static_cast<int>(std::round(px)), static_cast<int>(std::round(py)));
    };

    cv::line(plot, cv::Point(left, top), cv::Point(left, height - bottom), cv::Scalar(0, 0, 0), 2);
    cv::line(plot, cv::Point(left, height - bottom), cv::Point(width - right, height - bottom), cv::Scalar(0, 0, 0), 2);

    for (size_t i = 1; i < r_values.size(); ++i) {
        cv::line(plot, to_point(r_values[i - 1], zonal_mse[i - 1]), to_point(r_values[i], zonal_mse[i]), cv::Scalar(40, 90, 220), 2);
        cv::line(plot, to_point(r_values[i - 1], threshold_mse[i - 1]), to_point(r_values[i], threshold_mse[i]), cv::Scalar(50, 170, 80), 2);
    }
    for (size_t i = 0; i < r_values.size(); ++i) {
        cv::circle(plot, to_point(r_values[i], zonal_mse[i]), 5, cv::Scalar(40, 90, 220), cv::FILLED);
        cv::rectangle(plot, to_point(r_values[i], threshold_mse[i]) - cv::Point(4, 4), to_point(r_values[i], threshold_mse[i]) + cv::Point(4, 4), cv::Scalar(50, 170, 80), cv::FILLED);
    }

    cv::putText(plot, "MSE vs Retained Information", cv::Point(210, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(plot, "Zonal coding", cv::Point(570, 80), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(40, 90, 220), 2, cv::LINE_AA);
    cv::putText(plot, "Threshold coding", cv::Point(570, 110), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(50, 170, 80), 2, cv::LINE_AA);
    cv::putText(plot, "Retained Information r (%)", cv::Point(280, height - 22), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::putText(plot, "MSE", cv::Point(20, 45), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    cv::imwrite(path.string(), plot);
}

static bool is_showcase(int r_value)
{
    for (int showcase : SHOWCASE_R_VALUES) {
        if (showcase == r_value) {
            return true;
        }
    }
    return false;
}

int main()
{
    try {
        const fs::path root = fs::path("..");
        const fs::path image_path = find_file(root / "Images", "board.png");
        const fs::path output_dir = root / "outputs" / "askisi5_dct_compression_cpp";
        fs::create_directories(output_dir);

        const cv::Mat original = imread_grayscale_any_path(image_path);
        const cv::Size original_size(original.cols, original.rows);
        const cv::Mat padded = pad_to_block_size(original, BLOCK_SIZE);
        const cv::Mat coefficients = dct_blocks(padded);

        std::vector<int> r_values;
        std::vector<double> zonal_mse_values;
        std::vector<double> threshold_mse_values;
        std::vector<std::pair<int, cv::Mat>> zonal_showcase;
        std::vector<std::pair<int, cv::Mat>> threshold_showcase;

        std::ostringstream metrics_json;
        metrics_json << "{\n";
        metrics_json << "  \"block_size\": " << BLOCK_SIZE << ",\n";
        metrics_json << "  \"original_shape\": [" << original.rows << ", " << original.cols << "],\n";
        metrics_json << "  \"padded_shape\": [" << padded.rows << ", " << padded.cols << "],\n";
        metrics_json << "  \"metrics\": {\n";

        for (size_t idx = 0; idx < sizeof(R_VALUES) / sizeof(R_VALUES[0]); ++idx) {
            const int r_value = R_VALUES[idx];
            const int keep_count = std::max(1, static_cast<int>(std::round((r_value / 100.0) * (BLOCK_SIZE * BLOCK_SIZE))));

            const cv::Mat zonal_coeffs = apply_zonal(coefficients, keep_count);
            cv::Mat zonal_reconstructed = crop_original_size(inverse_dct_blocks(zonal_coeffs), original_size);
            cv::max(zonal_reconstructed, 0.0, zonal_reconstructed);
            cv::min(zonal_reconstructed, 255.0, zonal_reconstructed);
            const double zonal_error = mse(original, zonal_reconstructed);

            const cv::Mat threshold_coeffs = apply_threshold(coefficients, keep_count);
            cv::Mat threshold_reconstructed = crop_original_size(inverse_dct_blocks(threshold_coeffs), original_size);
            cv::max(threshold_reconstructed, 0.0, threshold_reconstructed);
            cv::min(threshold_reconstructed, 255.0, threshold_reconstructed);
            const double threshold_error = mse(original, threshold_reconstructed);

            r_values.push_back(r_value);
            zonal_mse_values.push_back(zonal_error);
            threshold_mse_values.push_back(threshold_error);

            cv::imwrite((output_dir / ("zonal_r_" + std::to_string(r_value) + ".png")).string(), clip_to_uint8(zonal_reconstructed));
            cv::imwrite((output_dir / ("threshold_r_" + std::to_string(r_value) + ".png")).string(), clip_to_uint8(threshold_reconstructed));

            if (is_showcase(r_value)) {
                zonal_showcase.push_back({r_value, zonal_reconstructed});
                threshold_showcase.push_back({r_value, threshold_reconstructed});
            }

            metrics_json << "    \"" << r_value << "\": {\"zonal_mse\": " << zonal_error << ", \"threshold_mse\": " << threshold_error << "}";
            metrics_json << (idx + 1 == sizeof(R_VALUES) / sizeof(R_VALUES[0]) ? "\n" : ",\n");
        }

        metrics_json << "  }\n";
        metrics_json << "}\n";

        save_comparison_figure(original, zonal_showcase, "Zonal", output_dir / "zonal_showcase.png");
        save_comparison_figure(original, threshold_showcase, "Threshold", output_dir / "threshold_showcase.png");
        save_mse_curve(r_values, zonal_mse_values, threshold_mse_values, output_dir / "mse_curve.png");

        std::ofstream summary_file(output_dir / "summary.json");
        summary_file << metrics_json.str();

        std::cout << metrics_json.str();
        std::cout << "Outputs saved to: " << output_dir.string() << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}

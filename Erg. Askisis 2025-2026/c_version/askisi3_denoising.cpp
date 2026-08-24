#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const double TARGET_SNR_DB = 15.0;
static const double IMPULSE_DENSITY = 0.20;
static const int KERNEL_SIZES[] = {3, 5};

static fs::path find_file(const fs::path &root, const std::string &filename)
{
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().filename().string() == filename) {
            return entry.path();
        }
    }
    throw std::runtime_error("Could not find input file: " + filename);
}

static cv::Mat imread_unit_any_path(const fs::path &path)
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
    decoded.convertTo(result, CV_64F, 1.0 / 255.0);
    return result;
}

static cv::Mat clip_unit(const cv::Mat &image)
{
    cv::Mat bounded;
    cv::max(image, 0.0, bounded);
    cv::min(bounded, 1.0, bounded);
    return bounded;
}

static cv::Mat unit_to_uint8(const cv::Mat &image)
{
    cv::Mat clipped = clip_unit(image);
    cv::Mat out;
    clipped.convertTo(out, CV_8U, 255.0);
    return out;
}

static double mse(const cv::Mat &reference, const cv::Mat &test)
{
    cv::Mat diff;
    cv::subtract(reference, test, diff);
    cv::multiply(diff, diff, diff);
    return cv::mean(diff)[0];
}

static double psnr(const cv::Mat &reference, const cv::Mat &test)
{
    const double error = mse(reference, test);
    if (error == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 10.0 * std::log10(1.0 / error);
}

static double snr_db(const cv::Mat &reference, const cv::Mat &test)
{
    cv::Mat noise = test - reference;
    cv::Mat signal_sq;
    cv::Mat noise_sq;
    cv::multiply(reference, reference, signal_sq);
    cv::multiply(noise, noise, noise_sq);

    const double signal_power = cv::mean(signal_sq)[0];
    const double noise_power = cv::mean(noise_sq)[0];
    if (noise_power == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 10.0 * std::log10(signal_power / noise_power);
}

static cv::Mat gaussian_noise_base(cv::Size size, std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    cv::Mat noise(size, CV_64F);
    for (int y = 0; y < noise.rows; ++y) {
        double *row = noise.ptr<double>(y);
        for (int x = 0; x < noise.cols; ++x) {
            row[x] = normal(rng);
        }
    }
    return noise;
}

static std::pair<cv::Mat, double> add_gaussian_noise_with_exact_snr(const cv::Mat &image, double target_snr_db, std::mt19937 &rng)
{
    cv::Mat base_noise = gaussian_noise_base(image.size(), rng);
    cv::Mat signal_sq;
    cv::multiply(image, image, signal_sq);

    const double signal_power = cv::mean(signal_sq)[0];
    const double target_noise_power = signal_power / std::pow(10.0, target_snr_db / 10.0);
    const double initial_scale = std::sqrt(target_noise_power);

    double low_scale = 0.0;
    double high_scale = std::max(initial_scale, 1e-12);
    cv::Mat noisy_high = clip_unit(image + high_scale * base_noise);

    while (snr_db(image, noisy_high) > target_snr_db) {
        high_scale *= 2.0;
        noisy_high = clip_unit(image + high_scale * base_noise);
    }

    cv::Mat best_noisy = noisy_high;
    for (int i = 0; i < 40; ++i) {
        const double mid_scale = 0.5 * (low_scale + high_scale);
        cv::Mat noisy_mid = clip_unit(image + mid_scale * base_noise);
        const double actual_snr = snr_db(image, noisy_mid);
        best_noisy = noisy_mid;

        if (actual_snr > target_snr_db) {
            low_scale = mid_scale;
        } else {
            high_scale = mid_scale;
        }
    }

    cv::Mat diff = best_noisy - image;
    cv::multiply(diff, diff, diff);
    return {best_noisy, cv::mean(diff)[0]};
}

static cv::Mat add_impulse_noise(const cv::Mat &image, double density, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    cv::Mat noisy = image.clone();

    for (int y = 0; y < noisy.rows; ++y) {
        double *row = noisy.ptr<double>(y);
        for (int x = 0; x < noisy.cols; ++x) {
            const double value = uniform(rng);
            if (value < density / 2.0) {
                row[x] = 0.0;
            } else if (value < density) {
                row[x] = 1.0;
            }
        }
    }

    return noisy;
}

static cv::Mat moving_average_filter(const cv::Mat &image, int kernel_size)
{
    cv::Mat filtered;
    cv::blur(image, filtered, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), cv::BORDER_REFLECT);
    return filtered;
}

static cv::Mat median_denoise(const cv::Mat &image, int kernel_size)
{
    cv::Mat image_32f;
    cv::Mat filtered_32f;
    cv::Mat filtered;
    image.convertTo(image_32f, CV_32F);
    cv::medianBlur(image_32f, filtered_32f, kernel_size);
    filtered_32f.convertTo(filtered, CV_64F);
    return filtered;
}

static cv::Mat panel(const cv::Mat &image, const std::string &title, int target_width = 330)
{
    cv::Mat color;
    cv::cvtColor(unit_to_uint8(image), color, cv::COLOR_GRAY2BGR);

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

static std::string metrics_json(const cv::Mat &reference, const cv::Mat &candidate)
{
    std::ostringstream out;
    out << "{\"mse\": " << mse(reference, candidate) << ", \"psnr_db\": " << psnr(reference, candidate) << "}";
    return out.str();
}

static std::string process_noise_case(const std::string &case_name, const cv::Mat &original, const cv::Mat &noisy, const fs::path &output_dir)
{
    std::ostringstream result;
    result << "    \"noisy_metrics\": " << metrics_json(original, noisy) << ",\n";
    result << "    \"filters\": {\n";

    std::vector<cv::Mat> comparison_panels;
    comparison_panels.push_back(panel(noisy, "Noisy Input"));

    for (size_t i = 0; i < sizeof(KERNEL_SIZES) / sizeof(KERNEL_SIZES[0]); ++i) {
        const int kernel_size = KERNEL_SIZES[i];
        const cv::Mat mean_image = moving_average_filter(noisy, kernel_size);
        const cv::Mat median_image = median_denoise(noisy, kernel_size);

        cv::imwrite((output_dir / (case_name + "_mean_" + std::to_string(kernel_size) + ".png")).string(), unit_to_uint8(mean_image));
        cv::imwrite((output_dir / (case_name + "_median_" + std::to_string(kernel_size) + ".png")).string(), unit_to_uint8(median_image));

        comparison_panels.push_back(panel(mean_image, "Mean " + std::to_string(kernel_size) + "x" + std::to_string(kernel_size)));
        comparison_panels.push_back(panel(median_image, "Median " + std::to_string(kernel_size) + "x" + std::to_string(kernel_size)));

        result << "      \"mean_" << kernel_size << "\": " << metrics_json(original, mean_image) << ",\n";
        result << "      \"median_" << kernel_size << "\": " << metrics_json(original, median_image);
        result << (i + 1 == sizeof(KERNEL_SIZES) / sizeof(KERNEL_SIZES[0]) ? "\n" : ",\n");
    }

    result << "    }\n";
    save_row_panel(comparison_panels, output_dir / (case_name + "_kernel_comparison.png"));
    return result.str();
}

int main()
{
    try {
        const fs::path root = fs::path("..");
        const fs::path image_path = find_file(root / "Images", "tiger.png");
        const fs::path output_dir = root / "outputs" / "askisi3_denoising_cpp";
        fs::create_directories(output_dir);

        std::mt19937 rng(42);
        const cv::Mat original = imread_unit_any_path(image_path);
        cv::imwrite((output_dir / "tiger_original.png").string(), unit_to_uint8(original));

        auto gaussian_pair = add_gaussian_noise_with_exact_snr(original, TARGET_SNR_DB, rng);
        const cv::Mat gaussian_noisy = gaussian_pair.first;
        const double gaussian_noise_power = gaussian_pair.second;
        const cv::Mat impulse_noisy = add_impulse_noise(original, IMPULSE_DENSITY, rng);

        cv::imwrite((output_dir / "tiger_gaussian_noisy.png").string(), unit_to_uint8(gaussian_noisy));
        cv::imwrite((output_dir / "tiger_impulse_noisy.png").string(), unit_to_uint8(impulse_noisy));

        auto combined_gaussian_pair = add_gaussian_noise_with_exact_snr(original, TARGET_SNR_DB, rng);
        const cv::Mat combined_noisy = add_impulse_noise(combined_gaussian_pair.first, IMPULSE_DENSITY, rng);
        const cv::Mat mean_then_median = median_denoise(moving_average_filter(combined_noisy, 3), 3);
        const cv::Mat median_then_mean = moving_average_filter(median_denoise(combined_noisy, 3), 3);

        cv::imwrite((output_dir / "combined_noisy.png").string(), unit_to_uint8(combined_noisy));
        cv::imwrite((output_dir / "combined_mean_then_median_restored.png").string(), unit_to_uint8(mean_then_median));
        cv::imwrite((output_dir / "combined_median_then_mean_restored.png").string(), unit_to_uint8(median_then_mean));

        save_row_panel({panel(original, "Original"), panel(combined_noisy, "Combined Noisy"), panel(mean_then_median, "Mean -> Median")}, output_dir / "combined_mean_then_median.png");
        save_row_panel({panel(original, "Original"), panel(combined_noisy, "Combined Noisy"), panel(median_then_mean, "Median -> Mean")}, output_dir / "combined_median_then_mean.png");

        std::ostringstream summary;
        summary << std::setprecision(8);
        summary << "{\n";
        summary << "  \"target_snr_db\": " << TARGET_SNR_DB << ",\n";
        summary << "  \"impulse_density\": " << IMPULSE_DENSITY << ",\n";
        summary << "  \"gaussian_noise_power\": " << gaussian_noise_power << ",\n";
        summary << "  \"gaussian_actual_snr_db\": " << snr_db(original, gaussian_noisy) << ",\n";
        summary << "  \"gaussian_case\": {\n" << process_noise_case("gaussian", original, gaussian_noisy, output_dir) << "  },\n";
        summary << "  \"impulse_case\": {\n" << process_noise_case("impulse", original, impulse_noisy, output_dir) << "  },\n";
        summary << "  \"combined_case\": {\n";
        summary << "    \"noise_power\": " << combined_gaussian_pair.second << ",\n";
        summary << "    \"noisy_metrics\": " << metrics_json(original, combined_noisy) << ",\n";
        summary << "    \"mean_then_median\": " << metrics_json(original, mean_then_median) << ",\n";
        summary << "    \"median_then_mean\": " << metrics_json(original, median_then_mean) << ",\n";
        summary << "    \"recommended_order\": \"" << (mse(original, median_then_mean) < mse(original, mean_then_median) ? "median_then_mean" : "mean_then_median") << "\"\n";
        summary << "  }\n";
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

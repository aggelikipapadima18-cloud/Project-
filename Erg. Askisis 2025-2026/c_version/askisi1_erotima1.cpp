#include <opencv2/opencv.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const int SELECTED_CUTOFF_RADIUS = 35;
static const int CUTOFF_CANDIDATES[] = {15, 25, 35, 45, 60};

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

static cv::Mat clip_to_uint8(const cv::Mat &image)
{
    cv::Mat bounded;
    cv::Mat clipped;
    cv::max(image, 0.0, bounded);
    cv::min(bounded, 255.0, bounded);
    bounded.convertTo(clipped, CV_8U);
    return clipped;
}

static cv::Mat linear_stretch(const cv::Mat &image)
{
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(image, &min_value, &max_value);

    if (max_value == min_value) {
        return cv::Mat::zeros(image.size(), CV_64F);
    }

    cv::Mat stretched;
    image.convertTo(stretched, CV_64F, 255.0 / (max_value - min_value), -255.0 * min_value / (max_value - min_value));
    return stretched;
}

static cv::Mat center_dft_origin(const cv::Mat &image)
{
    cv::Mat centered(image.size(), CV_64F);
    for (int y = 0; y < image.rows; ++y) {
        const double *src = image.ptr<double>(y);
        double *dst = centered.ptr<double>(y);
        for (int x = 0; x < image.cols; ++x) {
            dst[x] = src[x] * (((x + y) % 2 == 0) ? 1.0 : -1.0);
        }
    }
    return centered;
}

static cv::Mat dft2(const cv::Mat &image)
{
    cv::Mat spectrum;
    cv::dft(image, spectrum, cv::DFT_COMPLEX_OUTPUT);
    return spectrum;
}

static cv::Mat idft2_real(const cv::Mat &spectrum)
{
    cv::Mat restored;
    cv::dft(spectrum, restored, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return restored;
}

static cv::Mat magnitude_spectrum(const cv::Mat &spectrum)
{
    std::vector<cv::Mat> planes;
    cv::split(spectrum, planes);

    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    return magnitude;
}

static cv::Mat log1p_mat(const cv::Mat &image)
{
    cv::Mat shifted;
    cv::log(image + 1.0, shifted);
    return shifted;
}

static cv::Mat ideal_lowpass_filter(cv::Size size, double cutoff_radius)
{
    cv::Mat mask(size, CV_64F, cv::Scalar(0.0));
    const int center_y = size.height / 2;
    const int center_x = size.width / 2;

    for (int y = 0; y < size.height; ++y) {
        double *row = mask.ptr<double>(y);
        for (int x = 0; x < size.width; ++x) {
            const double distance = std::sqrt((y - center_y) * (y - center_y) + (x - center_x) * (x - center_x));
            row[x] = (distance <= cutoff_radius) ? 1.0 : 0.0;
        }
    }

    return mask;
}

static cv::Mat apply_mask_to_spectrum(const cv::Mat &spectrum, const cv::Mat &mask)
{
    std::vector<cv::Mat> mask_planes = {mask, mask};
    cv::Mat complex_mask;
    cv::merge(mask_planes, complex_mask);

    cv::Mat filtered;
    cv::multiply(spectrum, complex_mask, filtered);
    return filtered;
}

static double mse(const cv::Mat &reference, const cv::Mat &test)
{
    cv::Mat diff;
    cv::subtract(reference, test, diff);
    cv::multiply(diff, diff, diff);
    return cv::mean(diff)[0];
}

static cv::Mat panel(const cv::Mat &image, const std::string &title, bool color = false)
{
    cv::Mat display;
    if (color) {
        display = image.clone();
    } else if (image.type() == CV_8U) {
        cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    } else {
        cv::cvtColor(normalize_to_uint8(image), display, cv::COLOR_GRAY2BGR);
    }

    const int title_height = 42;
    cv::Mat canvas(display.rows + title_height, display.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    display.copyTo(canvas(cv::Rect(0, title_height, display.cols, display.rows)));
    cv::putText(canvas, title, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    return canvas;
}

static void save_row_panel(const std::vector<cv::Mat> &panels, const fs::path &path)
{
    cv::Mat combined;
    cv::hconcat(panels, combined);
    cv::imwrite(path.string(), combined);
}

static void save_mse_curve(const std::vector<int> &radii, const std::vector<double> &errors, const fs::path &path)
{
    const int width = 800;
    const int height = 500;
    const int left = 80;
    const int right = 30;
    const int top = 40;
    const int bottom = 70;

    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    const auto minmax_x = std::minmax_element(radii.begin(), radii.end());
    const auto minmax_y = std::minmax_element(errors.begin(), errors.end());
    const double min_x = static_cast<double>(*minmax_x.first);
    const double max_x = static_cast<double>(*minmax_x.second);
    const double min_y = *minmax_y.first;
    const double max_y = *minmax_y.second;
    const double y_range = std::max(max_y - min_y, 1e-12);

    auto to_point = [&](int radius, double error) {
        const double px = left + (radius - min_x) * (width - left - right) / std::max(max_x - min_x, 1.0);
        const double py = height - bottom - (error - min_y) * (height - top - bottom) / y_range;
        return cv::Point(static_cast<int>(std::round(px)), static_cast<int>(std::round(py)));
    };

    cv::line(plot, cv::Point(left, top), cv::Point(left, height - bottom), cv::Scalar(0, 0, 0), 2);
    cv::line(plot, cv::Point(left, height - bottom), cv::Point(width - right, height - bottom), cv::Scalar(0, 0, 0), 2);

    for (size_t i = 1; i < radii.size(); ++i) {
        cv::line(plot, to_point(radii[i - 1], errors[i - 1]), to_point(radii[i], errors[i]), cv::Scalar(40, 80, 200), 2);
    }
    for (size_t i = 0; i < radii.size(); ++i) {
        cv::circle(plot, to_point(radii[i], errors[i]), 5, cv::Scalar(40, 80, 200), cv::FILLED);
    }

    cv::putText(plot, "Lowpass Cutoff Radius vs Reconstruction MSE", cv::Point(110, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(plot, "Cutoff Radius", cv::Point(330, height - 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::putText(plot, "MSE", cv::Point(18, 45), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    cv::imwrite(path.string(), plot);
}

int main()
{
    try {
        const fs::path root = fs::path("..");
        const fs::path image_path = find_file(root / "Images", "moon.jpg");
        const fs::path output_dir = root / "outputs" / "askisi1_erotima1_cpp";
        fs::create_directories(output_dir);

        const cv::Mat original = imread_grayscale_any_path(image_path);
        const cv::Mat stretched = linear_stretch(original);
        const cv::Mat centered = center_dft_origin(stretched);
        const cv::Mat centered_visual = normalize_to_uint8(centered);
        const cv::Mat spectrum = dft2(centered);
        const cv::Mat linear_magnitude = magnitude_spectrum(spectrum);
        const cv::Mat log_magnitude = log1p_mat(linear_magnitude);

        std::vector<int> cutoff_radii;
        std::vector<double> mse_values;

        cv::Mat selected_lowpass_mask;
        cv::Mat selected_filtered_log_magnitude;
        cv::Mat selected_restored_spatial;
        cv::Mat selected_filtered_spectrum;
        double selected_mse = 0.0;

        for (int cutoff_radius : CUTOFF_CANDIDATES) {
            cv::Mat lowpass_mask = ideal_lowpass_filter(spectrum.size(), cutoff_radius);
            cv::Mat filtered_spectrum = apply_mask_to_spectrum(spectrum, lowpass_mask);
            cv::Mat filtered_log_magnitude = log1p_mat(magnitude_spectrum(filtered_spectrum));
            cv::Mat restored_centered = idft2_real(filtered_spectrum);
            cv::Mat restored_spatial = center_dft_origin(restored_centered);
            cv::max(restored_spatial, 0.0, restored_spatial);
            cv::min(restored_spatial, 255.0, restored_spatial);

            const double reconstruction_mse = mse(stretched, restored_spatial);
            cutoff_radii.push_back(cutoff_radius);
            mse_values.push_back(reconstruction_mse);

            if (cutoff_radius == SELECTED_CUTOFF_RADIUS) {
                selected_lowpass_mask = lowpass_mask;
                selected_filtered_spectrum = filtered_spectrum;
                selected_filtered_log_magnitude = filtered_log_magnitude;
                selected_restored_spatial = restored_spatial;
                selected_mse = reconstruction_mse;
            }
        }

        cv::imwrite((output_dir / "moon_stretched.png").string(), clip_to_uint8(stretched));
        cv::imwrite((output_dir / "moon_centered_visual.png").string(), centered_visual);
        cv::imwrite((output_dir / "dft_magnitude_linear.png").string(), normalize_to_uint8(linear_magnitude));
        cv::imwrite((output_dir / "dft_magnitude_log.png").string(), normalize_to_uint8(log_magnitude));
        cv::imwrite((output_dir / "lowpass_mask.png").string(), clip_to_uint8(selected_lowpass_mask * 255.0));
        cv::imwrite((output_dir / "filtered_spectrum_log.png").string(), normalize_to_uint8(selected_filtered_log_magnitude));
        cv::imwrite((output_dir / "moon_lowpass_restored.png").string(), clip_to_uint8(selected_restored_spatial));

        save_row_panel(
            {
                panel(clip_to_uint8(original), "Original"),
                panel(clip_to_uint8(stretched), "Linear Stretch"),
                panel(centered_visual, "Centered for DFT")
            },
            output_dir / "preprocessing_summary.png"
        );

        save_row_panel(
            {
                panel(normalize_to_uint8(linear_magnitude), "Magnitude - Linear Scale"),
                panel(normalize_to_uint8(log_magnitude), "Magnitude - Log Scale")
            },
            output_dir / "spectrum_summary.png"
        );

        save_row_panel(
            {
                panel(clip_to_uint8(selected_lowpass_mask * 255.0), "Ideal Lowpass Mask"),
                panel(normalize_to_uint8(selected_filtered_log_magnitude), "Filtered Spectrum - Log Scale"),
                panel(clip_to_uint8(selected_restored_spatial), "Restored Spatial Image")
            },
            output_dir / "filtering_summary.png"
        );

        save_mse_curve(cutoff_radii, mse_values, output_dir / "cutoff_mse_curve.png");

        std::ostringstream summary;
        double original_min = 0.0;
        double original_max = 0.0;
        double stretched_min = 0.0;
        double stretched_max = 0.0;
        cv::minMaxLoc(original, &original_min, &original_max);
        cv::minMaxLoc(stretched, &stretched_min, &stretched_max);

        summary << "Input image: " << image_path.string() << "\n";
        summary << "Original shape: (" << original.rows << ", " << original.cols << ")\n";
        summary << "Original min/max: " << original_min << " / " << original_max << "\n";
        summary << "Stretched min/max: " << stretched_min << " / " << stretched_max << "\n";
        summary << "Tested cutoff radii: 15, 25, 35, 45, 60\n";
        summary << "Selected cutoff radius: " << SELECTED_CUTOFF_RADIUS << "\n";
        summary << "Reconstruction MSE after lowpass filtering: " << selected_mse << "\n";
        summary << "Outputs saved to: " << output_dir.string() << "\n";

        std::ofstream summary_file(output_dir / "summary.txt");
        summary_file << summary.str();

        std::cout << summary.str();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}

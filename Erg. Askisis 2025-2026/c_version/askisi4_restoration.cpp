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

static const double TARGET_SNR_DB = 10.0;

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

static cv::Mat clip_255(const cv::Mat &image)
{
    cv::Mat bounded;
    cv::max(image, 0.0, bounded);
    cv::min(bounded, 255.0, bounded);
    return bounded;
}

static cv::Mat to_uint8(const cv::Mat &image)
{
    cv::Mat clipped = clip_255(image);
    cv::Mat out;
    clipped.convertTo(out, CV_8U);
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
    return 10.0 * std::log10((255.0 * 255.0) / error);
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
    cv::Mat noisy_high = clip_255(image + high_scale * base_noise);

    while (snr_db(image, noisy_high) > target_snr_db) {
        high_scale *= 2.0;
        noisy_high = clip_255(image + high_scale * base_noise);
    }

    cv::Mat best_noisy = noisy_high;
    for (int i = 0; i < 40; ++i) {
        const double mid_scale = 0.5 * (low_scale + high_scale);
        cv::Mat noisy_mid = clip_255(image + mid_scale * base_noise);
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

static double median_value(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }

    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double median = values[middle];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
        median = 0.5 * (median + values[middle - 1]);
    }
    return median;
}

static double estimate_noise_power(const cv::Mat &noisy)
{
    cv::Mat smooth;
    cv::GaussianBlur(noisy, smooth, cv::Size(0, 0), 1.5, 1.5, cv::BORDER_REFLECT);
    cv::Mat residual = noisy - smooth;

    std::vector<double> values;
    values.reserve(static_cast<size_t>(residual.total()));
    for (int y = 0; y < residual.rows; ++y) {
        const double *row = residual.ptr<double>(y);
        for (int x = 0; x < residual.cols; ++x) {
            values.push_back(row[x]);
        }
    }

    const double med = median_value(values);
    for (double &value : values) {
        value = std::abs(value - med);
    }
    const double mad = median_value(values);
    const double sigma_hat = mad / 0.6745;
    return std::max(sigma_hat * sigma_hat, 1e-12);
}

static cv::Mat wiener_filter_frequency(const cv::Mat &noisy, double noise_power)
{
    cv::Mat noisy_freq;
    cv::dft(noisy, noisy_freq, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes;
    cv::split(noisy_freq, planes);

    cv::Mat magnitude_sq;
    cv::magnitude(planes[0], planes[1], magnitude_sq);
    cv::multiply(magnitude_sq, magnitude_sq, magnitude_sq);
    cv::Mat noisy_psd = magnitude_sq / static_cast<double>(noisy.rows * noisy.cols);

    cv::Mat signal_psd;
    cv::max(noisy_psd - noise_power, 1e-12, signal_psd);
    cv::Mat gain = signal_psd / (signal_psd + noise_power);

    cv::multiply(planes[0], gain, planes[0]);
    cv::multiply(planes[1], gain, planes[1]);

    cv::Mat filtered_freq;
    cv::merge(planes, filtered_freq);

    cv::Mat restored;
    cv::dft(filtered_freq, restored, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return clip_255(restored);
}

static cv::Mat panel(const cv::Mat &image, const std::string &title, int target_width = 330)
{
    cv::Mat color;
    cv::cvtColor(to_uint8(image), color, cv::COLOR_GRAY2BGR);

    const double scale = static_cast<double>(target_width) / color.cols;
    cv::Mat resized;
    cv::resize(color, resized, cv::Size(target_width, std::max(1, static_cast<int>(color.rows * scale))));

    const int title_height = 40;
    cv::Mat canvas(resized.rows + title_height, resized.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    resized.copyTo(canvas(cv::Rect(0, title_height, resized.cols, resized.rows)));
    cv::putText(canvas, title, cv::Point(10, 27), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
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

static std::string matlab_helper_text(const fs::path &input_dir, const fs::path &output_dir)
{
    std::ostringstream text;
    text << "addpath('" << input_dir.generic_string() << "');\n";
    text << "I = im2double(imread('" << (input_dir / "new_york.png").generic_string() << "'));\n";
    text << "delta = zeros(size(I));\n";
    text << "delta(1,1) = 1;\n";
    text << "H_impulse = psf(delta);\n";
    text << "blurred = psf(I);\n";
    text << "save('" << (output_dir / "psf_export.mat").generic_string() << "', 'H_impulse', 'blurred');\n";
    text << "disp('Saved psf_export.mat');\n";
    return text.str();
}

int main()
{
    try {
        const fs::path root = fs::path("..");
        const fs::path image_path = find_file(root / "Images", "new_york.png");
        const fs::path psf_path = find_file(root / "Images", "psf.p");
        const fs::path input_dir = psf_path.parent_path();
        const fs::path output_dir = root / "outputs" / "askisi4_restoration_cpp";
        fs::create_directories(output_dir);

        std::mt19937 rng(123);
        const cv::Mat original = imread_grayscale_any_path(image_path);
        auto noisy_pair = add_gaussian_noise_with_exact_snr(original, TARGET_SNR_DB, rng);
        const cv::Mat noisy = noisy_pair.first;
        const double noise_power = noisy_pair.second;
        const double estimated_noise_power = estimate_noise_power(noisy);

        const cv::Mat known_noise_wiener = wiener_filter_frequency(noisy, noise_power);
        const cv::Mat unknown_noise_wiener = wiener_filter_frequency(noisy, estimated_noise_power);

        cv::imwrite((output_dir / "new_york_original.png").string(), to_uint8(original));
        cv::imwrite((output_dir / "new_york_noisy_snr10.png").string(), to_uint8(noisy));
        cv::imwrite((output_dir / "wiener_known_noise.png").string(), to_uint8(known_noise_wiener));
        cv::imwrite((output_dir / "wiener_unknown_noise.png").string(), to_uint8(unknown_noise_wiener));

        save_row_panel(
            {
                panel(original, "Original"),
                panel(noisy, "Noisy (SNR=10 dB)"),
                panel(known_noise_wiener, "Wiener - Known Noise"),
                panel(unknown_noise_wiener, "Wiener - Unknown Noise")
            },
            output_dir / "wiener_summary.png"
        );

        const fs::path helper_path = output_dir / "askisi4_export_psf_from_matlab.m";
        std::ofstream helper_file(helper_path);
        helper_file << matlab_helper_text(input_dir, output_dir);

        std::ostringstream summary;
        summary << std::setprecision(8);
        summary << "{\n";
        summary << "  \"target_snr_db\": " << TARGET_SNR_DB << ",\n";
        summary << "  \"noise_power\": " << noise_power << ",\n";
        summary << "  \"actual_snr_db\": " << snr_db(original, noisy) << ",\n";
        summary << "  \"estimated_noise_power_unknown_case\": " << estimated_noise_power << ",\n";
        summary << "  \"part_a_metrics\": {\n";
        summary << "    \"noisy\": {\"mse\": " << mse(original, noisy) << ", \"psnr_db\": " << psnr(original, noisy) << "},\n";
        summary << "    \"known_noise_wiener\": {\"mse\": " << mse(original, known_noise_wiener) << ", \"psnr_db\": " << psnr(original, known_noise_wiener) << "},\n";
        summary << "    \"unknown_noise_wiener\": {\"mse\": " << mse(original, unknown_noise_wiener) << ", \"psnr_db\": " << psnr(original, unknown_noise_wiener) << "}\n";
        summary << "  },\n";
        summary << "  \"part_b_status\": \"MATLAB psf.p needs MATLAB export; the C++ program writes the MATLAB export file.\",\n";
        summary << "  \"psf_export_file\": \"" << helper_path.string() << "\",\n";
        summary << "  \"psf_p_path\": \"" << psf_path.string() << "\"\n";
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

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static int run_executable(const std::string &name)
{
    std::cout << "\n=== Running " << name << " ===\n";
    std::cout.flush();
    const std::string command = "build\\" + name + ".exe";
    const int status = std::system(command.c_str());
    if (status != 0) {
        std::cerr << name << " failed with status " << status << "\n";
    }
    return status;
}

int main()
{
    const std::vector<std::string> programs = {
        "askisi1_erotima1",
        "askisi2_histogram_equalization",
        "askisi3_denoising",
        "askisi4_restoration",
        "askisi5_dct_compression",
        "askisi6_edges"
    };

    int failed = 0;
    for (const std::string &program : programs) {
        if (run_executable(program) != 0) {
            failed = 1;
        }
    }

    if (failed == 0) {
        std::cout << "\nAll C++ assignment programs finished successfully.\n";
    }
    return failed;
}

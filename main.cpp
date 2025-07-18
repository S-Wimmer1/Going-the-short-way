#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <utility> // std::pair
#include <array>
#include <algorithm>
#include <string>


// Constants
const int AZIMUTH_MIN = -180;
const int AZIMUTH_MAX = 540;
const int ELEVATION_MIN = 15;
const int ELEVATION_MAX = 165;
const double AZ_SPEED = 1.6;
const double EL_SPEED = 1.3;
#ifndef M_PI
#define M_PI 3.141592653589793238462643383
#endif


std::tuple<double, double, double> azel_to_cartesian(double az_deg, double el_deg);
std::tuple<double, double> cartesian_to_azel(double x, double y, double z);
double slerp_path_time(const std::vector<std::tuple<double, double>>& path);


double dot_product(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


// Waypoints
std::vector<std::tuple<int, int>> waypoints = {
                                            {0, 15},
                                            {350, 30},
                                            {180, 85}
};


// Coordinate transformation
std::tuple<double, double, double> azel_to_cartesian(double az_deg, double el_deg) {
    double az = az_deg * M_PI / 180.0;
    double el = el_deg * M_PI / 180.0;

    double x = cos(el) * cos(az);
    double y = cos(el) * sin(az);
    double z = sin(el);
    return std::make_tuple(x, y, z);
}

// Helper function
double shortest_angular_distance(double from, double to) {
    double delta = std::fmod((to - from + 540.0), 360.0) - 180.0;
    return std::abs(delta);
}

// Reverse coordinate transformation
std::tuple<double, double> cartesian_to_azel(double x, double y, double z) {
    double hyp = std::hypot(x, y); // sqrt(x^2 + y^2)

    double az = std::atan2(y, x) * 180 / M_PI;
    if (az < 0) az += 360.0;

    double el = std::atan2(z, hyp) * 180.0 / M_PI;

    return std::make_tuple(az, el);
}


// SLERP: Spherical Linear Interpolation
std::vector<std::tuple<double, double>> slerp_path(double start_az, double start_el, double end_az, double end_el, int steps = 1000) {
    auto [x0, y0, z0] = azel_to_cartesian(start_az, start_el);
    auto [x1, y1, z1] = azel_to_cartesian(end_az, end_el);

    std::array<double, 3> p0 = {x0, y0, z0};
    std::array<double, 3> p1 = {x1, y1, z1};

    // angle between vectors
    double omega = std::acos(std::clamp(dot_product(p0, p1), -1.0, 1.0));

    std::vector<std::tuple<double, double>> path;

    if (std::abs(omega) < 1e-6) {
        for (int i = 0; i < steps; ++i) 
            path.emplace_back(start_az, start_el);
        return path;
    }

    double sin_omega = std::sin(omega);

    for (int i = 0; i < steps; ++i) {
        double t = static_cast<double>(i) / (steps - 1);
        double factor0 = std::sin((1 - t) * omega) / sin_omega;
        double factor1 = std::sin(t * omega) / sin_omega;

        // interpolated points
        double x = factor0 * p0[0] + factor1 * p1[0];
        double y = factor0 * p0[1] + factor1 * p1[1];
        double z = factor0 * p0[2] + factor1 * p1[2];

        auto [az, el] = cartesian_to_azel(x, y, z);
        path.emplace_back(az, el);
    }

    return path;
}


// Time calculation for path
double movement_time(std::tuple<double, double> start, std::tuple<double, double> end) {
    double az_start = std::get<0>(start);
    double el_start = std::get<1>(start);
    double az_end = std::get<0>(end);
    double el_end = std::get<1>(end);

    double delta_az = shortest_angular_distance(az_start, az_end);
    double delta_el = std::abs(el_end - el_start);

    double time_az = delta_az / AZ_SPEED;
    double time_el = delta_el / EL_SPEED;

    return std::max(time_az, time_el);
}


// Total time
double slerp_path_time(const std::vector<std::tuple<double, double>>& path) {
    double total_time = 0.0;
    for (int i = 1; i < path.size(); ++i)
        total_time += movement_time(path[i-1], path[i]);
    return total_time;
}


// Individual axis movement
std::vector<std::pair<std::string, std::vector<std::tuple<double, double>>>> axis_aligned_paths(std::tuple<double, double> start, std::tuple<double, double> end) {
    std::vector<std::pair<std::string, std::vector<std::tuple<double, double>>>> paths;

    double az_s = std::get<0>(start), el_s = std::get<1>(start);
    double az_e = std::get<0>(end), el_e = std::get<1>(end);

    // AZ first
    auto az_path1 = slerp_path(az_s, el_s, az_e, el_s, 250);
    auto el_path1 = slerp_path(az_e, el_s, az_e, el_e, 250);
    std::vector<std::tuple<double, double>> az_el_path(az_path1);
    az_el_path.insert(az_el_path.end(), el_path1.begin(), el_path1.end());
    paths.emplace_back("AZ->EL", az_el_path);

    auto el_path2 = slerp_path(az_s, el_s, az_s, el_e, 250);
    auto az_path2 = slerp_path(az_s, el_e, az_e, el_e, 250);
    std::vector<std::tuple<double, double>> el_az_path(el_path2);
    el_az_path.insert(el_az_path.end(), az_path2.begin(), az_path2.end());
    paths.emplace_back("EL->AZ", el_az_path);

    return paths;
}


// Simultaneous axis movement
std::vector<std::tuple<double, double>> simultaneous_az_el_path(std::tuple<double, double> start, std::tuple<double, double> end, int steps = 500) {
    double az1 = std::get<0>(start), el1 = std::get<1>(start);
    double az2 = std::get<0>(end), el2 = std::get<1>(end);

    double delta_az = std::fmod((az2 - az1 + 540.0), 360.0) - 180.0;
    double az_end_adjusted = az1 + delta_az;

    std::vector<std::tuple<double, double>> path;
    path.reserve(steps);

    for (int i = 0; i < steps; ++i) {
        double t = static_cast<double>(i) / (steps - 1);
        double az = az1 + (az_end_adjusted - az1) * t;
        double el = el1 + (el2 - el1) * t;
        path.emplace_back(az, el);
    }

    return path;
}


// Path struct
    struct Path_Result {
        std::tuple<double, double> from;
        std::tuple<double, double> to;
        std::string path_type;
        double time_seconds;
        std::vector<std::tuple<double, double>> path;
    };


// Testing
    void print_path_result(const Path_Result& result) {
        std::cout << "From: (" << std::get<0>(result.from) << ", " << std::get<1>(result.from) << ") -> "
                << "To: (" << std::get<0>(result.to) << ", " << std::get<1>(result.to) << ")\n";
        std::cout << "Type: " << result.path_type << "\n";
        std::cout << "Time: " << result.time_seconds << " s\n";
        std::cout << "Path Points: " << result.path.size() << "\n";
        std::cout << "Sample: ";
        for (size_t i = 0; i < std::min(size_t(5), result.path.size()); ++i) {
            std::cout << "(" << std::get<0>(result.path[i]) << ", " << std::get<1>(result.path[i]) << ") ";
        }
        std::cout << "\n-----------------------------\n";
    }


int main() {

    std::vector<Path_Result> all_paths;

    for (size_t i = 0; i < waypoints.size(); ++i) {
        auto start = waypoints[i];
        auto end = waypoints[i + 1];

        //SLERP
        auto path_slerp = slerp_path(
            std::get<0>(start), std::get<1>(start),
            std::get<0>(end), std::get<1>(end), 500
        );
        double time_slerp = slerp_path_time(path_slerp);

        Path_Result result{start, end, "SLERP", std::round(time_slerp * 100.0) / 100.0, path_slerp};
        all_paths.push_back(result);

        // Axis-aligned
        auto axis_paths = axis_aligned_paths(start, end);
        for (const auto& [name, path] : axis_paths) {
            double time_axis = slerp_path_time(path);
            all_paths.push_back({start, end, name, std::round(time_axis * 100.0) / 100.0, path});
        }

        // Simultaneous
        auto path_sim = simultaneous_az_el_path(start, end, 500);
        double time_sim = slerp_path_time(path_sim);
        all_paths.push_back({start, end, "AZ-EL Simult", std::round(time_sim * 100.0) / 100.0, path_sim});
    }

    // Testing
    for (const auto& path_result : all_paths)
        print_path_result(path_result);;

    return 0;
}

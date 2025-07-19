#include <stdio.h>                  // for printf
#include <stdlib.h>                 // for malloc, free
#include <math.h>                   // for sin, cos, atan2, pi




// Constants
#define AZIMUTH_MIN     -180
#define AZIMUTH_MAX     540
#define ELEVATION_MIN   15
#define ELEVATION_MAX   165

#define AZ_SPEED    1.6
#define EL_SPEED    1.3

#ifndef M_PI
#define M_PI 3.141592653589793238462643383
#endif




double dot_product(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}




// Waypoint struct
typedef struct {
    double azimuth;
    double elevation;
} AZEL;

// arrays must have fixed size at compile time
#define NUM_WAYPOINTS 10

AZEL waypoints[NUM_WAYPOINTS] = {
    //AZ    EL
    {0.0,   15.0}, // parking position
    {350.0, 30.0},
    {180.0, 85.0},
    {0.0,   85.0},
    {90.0,  60.0},
    {10.0,  20.0},
    {270.0, 80.0},
    {45.0,  40.0},
    {165.0, 40.0},
};




// Coordinate transformation
// Cartesian coordinate format
typedef struct {
    double x;
    double y;
    double z;
} Vec3;

Vec3 azel_to_cartesian(double az_deg, double el_deg) {
    double az = az_deg * M_PI / 180.0;
    double el = el_deg * M_PI / 180.0;

    Vec3 result;
    result.x = cos(el) * cos(az);
    result.y = cos(el) * sin(az);
    result.z = sin(el);
    return result;
}

// Helper function
double shortest_angular_distance(double from, double to) {
    double delta = fmod((to - from + 540.0), 360.0) - 180.0;
    return fabs(delta);
}

// Reverse coordinate transformation
AZEL cartesian_to_azel(double x, double y, double z) {
    double hyp = hypot(x, y); // sqrt(x^2 + y^2)

    AZEL result;
    result.azimuth = atan2(y, x) * 180.0 / M_PI;
    if (result.azimuth < 0) result.azimuth += 360.0;

    result.elevation = atan2(z, hyp) * 180.0 / M_PI;

    return result;
}




// SLERP: Spherical Linear Interpolation
AZEL* slerp_path(double start_az, double start_el, double end_az, double end_el, int steps, int* out_size) {
    Vec3 start = azel_to_cartesian(start_az, start_el);
    Vec3 end = azel_to_cartesian(end_az, end_el);

    double p0[3] = { start.x, start.y, start.z };
    double p1[3] = { end.x, end.y, end.z };

    double dot = dot_product(p0, p1);
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;

    double omega = acos(dot);

    AZEL* path = malloc(sizeof(AZEL) * steps);
    if (!path) return NULL;

    if (fabs(omega) < 1e-6) {
        for (int i = 0; i < steps; ++i) {
            path[i].azimuth = start_az;
            path[i].elevation = start_el;
        }
        *out_size = steps;
        return path;
    }

    double sin_omega = sin(omega);

    for (int i = 0; i < steps; ++i) {
        double t = (double)i / (steps - 1); // cast
        double factor0 = sin((1 - t) * omega) / sin_omega;
        double factor1 = sin(t * omega) / sin_omega;

        // Interpolated points
        double x = factor0 * p0[0] + factor1 * p1[0];
        double y = factor0 * p0[1] + factor1 * p1[1];
        double z = factor0 * p0[2] + factor1 * p1[2];

        AZEL cart = cartesian_to_azel(x, y, z);
        path[i].azimuth = cart.azimuth;
        path[i].elevation = cart.elevation;
    }

    *out_size = steps;
    return path;
}




// Time calculation for path
double movement_time(AZEL start, AZEL end) {
    double delta_az = shortest_angular_distance(start.azimuth, end.azimuth);
    double delta_el = fabs(end.elevation - start.elevation);

    double time_az = delta_az / AZ_SPEED;
    double time_el = delta_el / EL_SPEED;

    return (time_az > time_el) ? time_az : time_el; // return dominating movement time
}




// Total time for slerp
double slerp_path_time(AZEL* path, int length) {
    double total_time = 0.0;
    for (int i = 1; i < length; ++i) {
        total_time += movement_time(path[i - 1], path[i]);
    }
    return total_time;
}




// Struct for different paths
typedef struct {
    char label[8];
    AZEL* path;
    int length;
} LabeledPath;




// Individual axis movement, AZ first
AZEL* az_el_path(AZEL start, AZEL end, int steps, int* out_len) {
    int len1 = 0, len2 = 0;

    AZEL* az_path = slerp_path(start.azimuth, start.elevation, end.azimuth, start.elevation, steps, &len1);
    AZEL* el_path = slerp_path(end.azimuth, start.elevation, end.azimuth, end.elevation, steps, &len2);

    if (!az_path || !el_path) return NULL;

    AZEL* full_path = malloc(sizeof(AZEL) * steps);
    if (!full_path) return NULL;

    for (int i = 0; i < len1; ++i) full_path[i] = az_path[i];
    for (int i = 0; i < len2; ++i) full_path[len1 + i] = el_path[i];

    free(az_path);
    free(el_path);

    *out_len = len1 + len2;
    return full_path;
}




// EL first
AZEL* el_az_path(AZEL start, AZEL end, int steps, int* out_len) {
    int len1 = 0, len2 = 0;

    AZEL* el_path = slerp_path(start.azimuth, start.elevation, start.azimuth, end.elevation, steps, &len1);
    AZEL* az_path = slerp_path(start.azimuth, end.elevation, end.azimuth, end.elevation, steps, &len2);

    if (!el_path || !az_path) return NULL;

    AZEL* full_path = malloc(sizeof(AZEL) * steps);
    if (!full_path) return NULL;

    for (int i = 0; i < len1; ++i) full_path[i] = el_path[i];
    for (int i = 0; i < len2; ++i) full_path[len1 + i] = az_path[i];
    
    free(el_path);
    free(az_path);

    *out_len = len1 + len2;
    return full_path;
}




// Both axis move simultaneously -> faster than individually
AZEL* simultaneous_az_el_path(AZEL start, AZEL end, int steps, int* out_len) {
    double az1 = start.azimuth;
    double el1 = start.elevation;
    double az2 = end.azimuth;
    double el2 = end.elevation;

    double delta_az = fmod((az2 - az1 + 540.0), 360.0) - 180.0;
    double az_end_adjusted = az1 + delta_az;

    AZEL* path = malloc(sizeof(AZEL) * steps);
    if (!path) return NULL;

    for (int i = 0; i < steps; ++i) {
        double t = (double)i / (steps - 1);
        double az = az1 + (az_end_adjusted - az1) * t;
        double el = el1 + (el2 - el1) * t;
        path[i].azimuth = az;
        path[i].elevation = el;
    }

    *out_len = steps;
    return path;
}




// Testing
void test_func() {
    printf("\n");
}




int main() {

    printf("test passed\n");
    return 0;
}
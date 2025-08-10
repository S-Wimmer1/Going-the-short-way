import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd # Dataframe
import seaborn as sns # Boxplot




# Constants (telescope constraint and axis-speeds)
AZIMUTH_MIN     = -180 # "under"-rotation
AZIMUTH_MAX     = 540 # with over-rotation
ELEVATION_MIN   = 15 # balcony ground collision at <=14°
ELEVATION_MAX   = 165 # flip-over
AZ_SPEED        = 1.6 # degrees per second, practically tested
EL_SPEED        = 1.3




# Waypoints for simulation
waypoints = [
    #AZ     EL
    (0,     15), # parking position
    (350,   30),
    (180,   85),
    (0,     85),
    (90,    60),
    (10,    20),
    (270,   80),
    (45,    40),
    (165,   40),
]




# Accurate movement with great circle works with cartesian 3D vectors
def azel_to_cartesian(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    # standard unit spherical coordinates
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])




# Reverse coordinate transformation
def cartesian_to_azel(vec: np.ndarray) -> Tuple[float, float]:
    x, y, z = vec
    hyp = np.hypot(x, y) # hypotenuse = sqrt(x**2 + y**2)
    az = np.degrees(np.arctan2(y, x)) % 360 # full-circle azimuth
    el = np.degrees(np.arctan2(z, hyp))
    return az, el




# For polar projection plot
def azel_to_polar(az_deg: float, el_deg: float) -> Tuple[float, float]:
    if el_deg > 90:
        el_deg = 180 - el_deg # flip over adjusting
    theta = np.radians(az_deg % 360)
    radius = 90 - el_deg # center = el_90
    return theta, radius




# SLERP: Spherical Linear Interpolation
def slerp_path(start_az: float, start_el: float, end_az: float, end_el: float, steps: int = 1000) -> List[Tuple[float, float]]:
    # convert to 3D vectors
    p0 = azel_to_cartesian(start_az, start_el)
    p1 = azel_to_cartesian(end_az, end_el)

    # angle between start/end vector in radians
    # dot product of 2 unit vectors => cos(omega) (since theta will be used for polar)
    omega = np.arccos(np.clip(np.dot(p0, p1), -1.0, 1.0)) # ensure it stays between -1 and 1

    # small angles ~ linear anyway
    if np.isclose(omega, 0):
        return [(start_az, start_el)] * steps

    # otherwise calculate steps along the path
    sin_omega = np.sin(omega)
    path = []

    # formula: point(t) = [sin((1-t)*omega)*p0 + sin(t*omega)*p1]/sin(omega) -> sin() = weights
    # no weights -> cut through circle, like |x| + |y| + |z|
    # interpolate along spherical arc
    for t in np.linspace(0, 1, steps):
        factor0 = np.sin((1-t) * omega) / sin_omega
        factor1 = np.sin(t * omega) / sin_omega
        pt = factor0 * p0 + factor1 * p1
        az, el = cartesian_to_azel(pt) # reverse coordinate after calculation of path
        path.append((az, el))
    return path




# Time calculation for a path (NOTE: no curved paths!!!)
def movement_time(start: Tuple[float, float], end: Tuple[float, float], sequential: bool = False) -> float:
    # Extract coordinates
    az_start, el_start = start
    az_end, el_end = end

    # Shortest distances for AZ/EL
    delta_az = min(abs(az_end - az_start), 360 - abs(az_end - az_start))
    delta_el = abs(el_end - el_start)

    time_az = delta_az / AZ_SPEED
    time_el = delta_el / EL_SPEED

    if sequential:
        return time_az + time_el # Separate movement
    else:
        return max(time_az, time_el) # Simultaneous AZ/EL movement




# Uses the SLERPs individual linear paths to calculate total time (approx curved)
def slerp_path_time(path: List[Tuple[float, float]], sequential: bool = False) -> float:
    total_time = 0.0
    for i in range(1, len(path)):
        total_time += movement_time(path[i-1], path[i], sequential)
    return total_time




# SIMULTANEOUS PATH WITHOUT OVER/UNDERROTATION OR FLIP-OVER
def simultaneous_az_el_path(start: Tuple[float, float], end: Tuple[float, float], steps: int = 500) -> List[Tuple[float, float]]:
    az1, el1 = start
    az2, el2 = end

    az_vals = np.linspace(az1, az2, steps)
    el_vals = np.linspace(el1, el2, steps)

    az_vals = np.mod(az_vals, 360.0)

    return list(zip(az_vals, el_vals))




#NOTE SIMULT AZEL WITH OVER/UNDERROTATION AND FLIP-OVER

# Gives angular difference in signed 180 range, thus showing direction [-180.0, 180.0]
def wrap180(az_diff):
    return (az_diff + 180.0) % 360.0 - 180.0




# Returns shortest rotation, positive -> clockwise, negative -> widdershins
def delta_shortest(az1, az2):
    return wrap180(az2 - az1)




# Calculates runtime for slewing telescope
def time_simult(az_abs_diff: float, el_abs_diff: float) -> float:
    t_az = az_abs_diff / AZ_SPEED
    t_el = el_abs_diff / EL_SPEED
    return max(t_az, t_el)




# Constructs straight path for plotting
def straight_path(start: Tuple[float, float], az_diff: float, el_diff: float, steps: int = 500) -> List[Tuple[float, float]]:
    az1, el1 = start
    t = np.linspace(0.0, 1.0, steps)
    az = az1 + az_diff * t
    el = el1 + el_diff * t
    return list(zip(az, el))




# Neither flip-over nor over/underrotate
def normal_path(start: Tuple[float, float], end: Tuple[float, float], steps: int = 500) -> dict:
    az1, el1 = start
    az2, el2 = end
    az_diff = delta_shortest(az1, az2)
    el_diff = el2 - el1
    return {
        "name": "no_flip_no_overr",
        "diffs": (az_diff, el_diff),
        "time": time_simult(abs(az_diff), abs(el_diff)),
        "path": straight_path(start, az_diff, el_diff, steps)
    }




def overrotate(start: Tuple[float, float], end: Tuple[float, float], steps: int = 500) -> dict:
    az1, el1 = start
    az2, el2 = end
    az_diff = (az2 - az1) % 360.0
    el_diff = el2 - el1
    path = straight_path(start, az_diff, el_diff, steps)
    path = [(az % 360.0, el) for az, el in path]
    return {
        "name": "overrotate",
        "diffs": (az_diff, el_diff),
        "time": time_simult(abs(az_diff), abs(el_diff)),
        "path": path
    }




def underrotate(start: Tuple[float, float], end: Tuple[float, float], steps: int = 500) -> dict:
    az1, el1 = start
    az2, el2 = end
    az_diff = -((az2 - az1) % 360.0)
    el_diff = el2 - el1
    path = straight_path(start, az_diff, el_diff, steps)
    path = [(az % 360.0, el) for az, el in path]
    return {
        "name": "underrotate",
        "diffs": (az_diff, el_diff),
        "time": time_simult(abs(az_diff), abs(el_diff)),
        "path": path
    }




def flip_over(start: Tuple[float, float], end: Tuple[float, float], steps: int = 500) -> dict:
    az1, el1 = start
    az2, el2 = end
    az_diff = az2 - az1 # not wrapped
    el_diff = el2 - el1
    path = straight_path(start, az_diff, el_diff, steps)
    # Put coordinates back in 0-360
    path = [(az % 360.0, el) for az, el in path]
    return {
        "name": "flip-over",
        "diffs": (az_diff, el_diff),
        "time": time_simult(abs(az_diff), abs(el_diff)),
        "path": path
    }




# UPGRADED LOGIC
def choose_path_logic(start: Tuple[float, float],
                         end: Tuple[float, float],
                         steps: int = 500,
                         az_flip_thresh: float = 150.0,
                         el_flip_thresh: float = 70.0) -> dict:
    az1, el1 = start
    az2, el2 = end

    
    # Determine path baseline values
    az_diff_signed = delta_shortest(az1, az2)

    # Decide if flip-over is useful
    near_zenith = (el1 >= el_flip_thresh) or (el2 >= el_flip_thresh)
    big_az = abs(az_diff_signed) >= az_flip_thresh


    if near_zenith and big_az:
        #TODO flip_over(az1, el1, az2, el2, steps)
        return flip_over(start, end, steps)

    elif (az1 + az_diff_signed > 360.0) and (az1 + az_diff_signed <= 540.0):
        #TODO overrotate(az1, el1, az2, el2, steps)
        return overrotate(start, end, steps)

    elif (az1 + az_diff_signed < 0.0) and (az1 + az_diff_signed >= -180.0):
        #TODO underrotate(az1, el1, az2, el2, steps)
        return underrotate(start, end, steps)

    else:
        # Normal (constrained by limits) movement
        return normal_path(start, end, steps)




def simult_azel_path_smart(start: Tuple[float, float],
                         end: Tuple[float, float],
                         steps: int = 500,
                         az_flip_thresh: float = 150.0,
                         el_flip_thresh: float = 70.0) -> dict:
    plan = choose_path_logic(start, end, steps, az_flip_thresh, el_flip_thresh)
    return plan["path"]




def random_waypoints(n: int = 10) -> List[Tuple[float, float]]:
    az = np.random.uniform(0, 360, n)
    el = np.random.uniform(ELEVATION_MIN, 90.0, n)

    return list(zip(az, el))




def compare_method(wps: List[Tuple[float, float]], steps: int = 500) -> pd.DataFrame:
    rows = []
    for i in range(len(wps) - 1):
        start, end = wps[i], wps[i+1]
        
        path_basic = simultaneous_az_el_path(start, end, steps=steps)
        time_basic = slerp_path_time(path_basic)
        rows.append({"From": start, "To": end, "Method": "AZ+EL Simult.", "Time (s)": time_basic})

        path_upgraded = simult_azel_path_smart(start, end, steps=steps)
        time_upgraded = slerp_path_time(path_upgraded)
        rows.append({"From": start, "To": end, "Method": "AZ+EL Upgraded", "Time (s)": time_upgraded})
    df = pd.DataFrame(rows)
    # Rounding
    df["From"] = df["From"].apply(lambda t: (round(t[0], 1), round(t[1], 1)))
    df["To"] = df["To"].apply(lambda t: (round(t[0], 1), round(t[1], 1)))

    return df




df_predet = compare_method(waypoints, steps=500)

N = 1000
wps = random_waypoints(N)
df_rand = compare_method(wps, steps=500)

df_all = pd.concat([df_predet.assign(Source="Determ."),
                    df_rand.assign(Source=f"Random {N}")],
                    ignore_index=True)




fastest = df_all.loc[df_all.groupby(["From", "To", "Source"])["Time (s)"].idxmin()]
win_counts = fastest.groupby(["Source", "Method"]).size().reset_index(name="Wins")
avg_times =df_all.groupby(["Source", "Method"])["Time (s)"].mean().reset_index()

print("\nFastest per segment (sample):")
print(fastest.head(10))

print("\nWin counts by method:")
print(win_counts)

print("\nAverage time by method:")
print(avg_times)

# Save CSV
# det_csv = "/mnt/data/deterministic_compare.csv"
# rand_csv = "/mnt/data/random_compare.csv"
# df_predet.to_csv(det_csv, index=False)
# df_rand.to_csv(rand_csv, index=False)



#NOTE PLOTS
# Line plot to show difference between basic and upgraded logic
plt.figure(figsize=(10, 6))
for method in df_predet["Method"].unique():
    y = df_predet[df_predet["Method"] == method]["Time (s)"].to_numpy()
    x = np.arange(len(y))
    plt.plot(x, y, label=method, alpha=0.8)
plt.title("Time per segment")
plt.xlabel("Segment Index")
plt.ylabel("Time (s)")
plt.grid(True)
plt.legend()
plt.show()




plt.figure(figsize=(8, 6))
data = [df_rand[df_rand["Method"] == method]["Time (s)"].to_numpy() for method in df_rand["Method"].unique()]
plt.boxplot(data, labels=list(df_rand["Method"].unique()), vert=True, patch_artist=False)
plt.title(f"Random {N} Waypoints: Time Distribution by Methods")
plt.ylabel("Time (s)")
plt.grid(True)
plt.show()





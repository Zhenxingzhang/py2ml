import numpy as np
import math
import gmplot
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

SW_lat = 52.464011  # (Latitude)
SW_lon = 13.274099  # (Longitude)
EARTH_RADIUS = 6371000  # Earth radius in meter

GATE_GPS = [52.516288, 13.377689]

BERLIN_GPS = [52.5167, 13.3833]

# The spree river could be represented with line segments between the following coordinates
SPREE_GPS = [
    [52.529198, 13.274099],
    [52.531835, 13.29234],
    [52.522116, 13.298541],
    [52.520569, 13.317349],
    [52.524877, 13.322434],
    [52.522788, 13.329],
    [52.517056, 13.332075],
    [52.522514, 13.340743],
    [52.517239, 13.356665],
    [52.523063, 13.372158],
    [52.519198, 13.379453],
    [52.522462, 13.392328],
    [52.520921, 13.399703],
    [52.515333, 13.406054],
    [52.514863, 13.416354],
    [52.506034, 13.435923],
    [52.496473, 13.461587],
    [52.487641, 13.483216],
    [52.488739, 13.491456],
    [52.464011, 13.503386]]

SATELLITE_START_GPS = [52.590117, 13.39915]
SATELLITE_END_GPS = [52.437385, 13.553989]


def gps_2_xy(p_lat, p_lon):
    """
    Convert GPS coordinates into XY coordinates
    :param p_lat: location latitude
    :param p_lon: location longitude
    :return: XY coordinates (x, y)
    """

    p_x = (p_lon - SW_lon) * math.cos(SW_lat * math.pi / 180) * 111.323
    p_y = (p_lat - SW_lat) * 111.323

    return p_x, p_y


def xy_2_gps(x, y):
    """
    convert XY coordinates back to GPS coordinates
    :param x:
    :param y:
    :return: GPS coordinates (latitude, longitude)
    """

    p_lon= + SW_lon - x /(111.323 * math.cos(SW_lat))
    p_lat= (y / 111.323) + SW_lat

    return round(p_lat, 6),round(p_lon, 6)


def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z


def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)


def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return X-x, Y-y, Z-z


def unit(v):
    x,y,z = v
    mag = length(v)
    return x/mag, y/mag, z/mag


def distance(p0,p1):
    return length(vector(p0,p1))


def scale(v,sc):
    x,y,z = v
    return x * sc, y * sc, z * sc


def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return x+X, y+Y, z+Z


def distance_to_linesegment(pnt, start, end):
    """
    Calculate the shortest distance between a point and a line segment.
    :param pnt: point in (x,y,z)
    :param start: line segment start point (x, y,z)
    :param end: line segment end point (x, y,z)
    :return: the shortest distance in float
    """
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    return round(dist,6)

def distance_ratio(radius):
    """
    used to convert the distance in earth surface to the distance in map
    :param radius: earth radius
    :return: the convert rate in float
    """
    degree = 1
    arc_length = degree * math.pi * radius / 180.0
    x, y = gps_2_xy(SW_lat+degree, SW_lon)

    return round(arc_length/y, 3)

def spree_distribution(x,y):
    """
    the density function for the normal distribution around the spree river
    :param x: location x
    :param y: location y
    :return: probability density in float
    """
    mu = 0.0
    sigma = (2730/distance_ratio(EARTH_RADIUS)) / 1.959963

    spree_coords = []
    for GPS_coord in SPREE_GPS:
        P_x, P_y = gps_2_xy(GPS_coord[0], GPS_coord[1])
        spree_coords.append([P_x, P_y, 0])
    distances = []

    for idx in range(len(spree_coords)-1):
        start_point = spree_coords[idx]
        end_point = spree_coords[idx+1]
        distances.append(distance_to_linesegment([x,y,0], start_point, end_point))
    shortest_distance = min(distances)
    return stats.norm.pdf(shortest_distance, loc=mu, scale=sigma)


def gate_distribution(x, y):
    """
    the density function for the log normal distribution around the spree river
    :param x: location x
    :param y: location y
    :return: probability density in float
    """

    mean = 4700 / distance_ratio(EARTH_RADIUS)
    mode = 3877 / distance_ratio(EARTH_RADIUS)

    GATE_coord = gps_2_xy(GATE_GPS[0], GATE_GPS[1])
    shortest_distance = length(vector([GATE_coord[0], GATE_coord[1],0], [x, y, 0]))

    mu = (2.0 * math.log(mean) + math.log(mode))/3
    sigma = math.sqrt((math.log(mean)-math.log(mode)) * 2.0 /3)

    return stats.lognorm.pdf(shortest_distance, s=sigma, loc=0, scale=math.exp(mu))


def satellite_distribution(x, y):
    """
    the density function for the log normal distribution along the satellite path.
    :param x: location x
    :param y: location y
    :return: probability density in float
    """

    mu = 0.0
    sigma = (2400 / distance_ratio(EARTH_RADIUS)) / 1.959963

    start_coord = gps_2_xy(SATELLITE_START_GPS[0], SATELLITE_START_GPS[1])
    end_coord = gps_2_xy(SATELLITE_END_GPS[0], SATELLITE_END_GPS[1])

    shortest_distance = distance_to_linesegment([x,y,0], [start_coord[0], start_coord[1], 0], [end_coord[0], end_coord[1], 0])
    return stats.norm.pdf(shortest_distance, loc=mu, scale=sigma)


def mixture_distribution(x, y):
    """
    the mixture density function by linear combining the three distributions
    :param x: location x
    :param y: location y
    :return: probability density in float
    """
    return 1.0/3 * spree_distribution(x, y) + 1.0/3 * gate_distribution(x, y) + 1.0/3 * satellite_distribution(x, y)


def objective_function(x):
    return -mixture_distribution(x[0],x[1])

def convert_spree(spree_coords):
    """
    A help method to convert the spree river's GPS coordinates into XY coordinates
    :param spree_coords: 2D array of GPS coordinates
    :return: 2D array of XY coordinates
    """
    XY_coords = []
    for GPS_coord in spree_coords:
        P_x, P_y = gps_2_xy(GPS_coord[0], GPS_coord[1])
        XY_coords.append([P_x, P_y])

    return XY_coords

def pds_in_grid(pdf):
    """
    A help method to generate a 2D dense location grid and calculate the corresponding mixture probability density values
    :param pdf: the mixture probability density function
    :return: location X, Y and the corresponding probability density values
    """
    x = np.linspace(0.0, 20.0, num=100)
    y = np.linspace(0.0, 12.0, num=75)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.ravel(), Y.ravel()]).T

    pds = np.zeros(7500)

    idx=0
    for point in XY:
        pds[idx] = pdf(point[0], point[1])
        idx = idx+1

    pds = pds.reshape((75, 100))
    return X,Y,pds


def find_maxima():
    """
    Use the gradient descent algorithm to find the local maxima value by given the starting point
    :return: the local gradient locations and their corresponding geo locations.
    """
    peak_point1 = optimize.minimize(objective_function, [15.02, 2.86], method='Nelder-Mead').x
    peak_point2 = optimize.minimize(objective_function, [3.12, 6.52], method='Nelder-Mead').x
    peak_point3 = optimize.minimize(objective_function, [11.50, 5.4], method='Nelder-Mead').x

    peak_points = [peak_point1, peak_point2, peak_point3]

    peak_coords = [xy_2_gps(peak_point1[0],peak_point1[1]),
                   xy_2_gps(peak_point2[0], peak_point2[1]),
                   xy_2_gps(peak_point3[0], peak_point3[1])]

    return peak_points, peak_coords


def draw_2D_plot(pdf, peak_points = []):
    """
    A help method to draw the pdf in 2D space.
    :param pdf: probability density function
    :param peak_points: the local maxima value, default value is empty
    :return:
    """

    X,Y,pds = pds_in_grid(pdf)

    CS = plt.contour(X, Y, pds)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    spree_coords = convert_spree(SPREE_GPS)
    spree_coords = np.asarray(spree_coords)
    plt.plot(spree_coords[:, 0], spree_coords[:, 1], '-.')

    GATE_coord = gps_2_xy(GATE_GPS[0], GATE_GPS[1])
    plt.plot(GATE_coord[0], GATE_coord[1], 'gD')

    start_coord = gps_2_xy(SATELLITE_START_GPS[0], SATELLITE_START_GPS[1])
    end_coord = gps_2_xy(SATELLITE_END_GPS[0], SATELLITE_END_GPS[1])
    satellite_path = np.asarray([start_coord, end_coord])
    plt.plot(satellite_path[:, 0], satellite_path[:, 1],color='y', ls='--', lw=1)

    for peak_point in peak_points:
        plt.plot(peak_point[0], peak_point[1], 'ro')

    plt.axis([-1, 20, -1, 20])

    plt.title('PDF')
    plt.axis('tight')
    plt.show()

def draw_3D_plot(pdf):
    """
    A help method to draw the pdf in 3D space.
    """

    X,Y,pds = pds_in_grid(pdf)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, pds, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(0, 0.50)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def draw_on_google_map(map_center, peak_coords):
    """
    A help method to draw an easy to read map and save it html.
    :param map_center: GPS coordinates of the map center
    :param peak_coords: GPS coordinates of the interesting locations
    :return:
    """

    gmap = gmplot.GoogleMapPlotter(map_center[0], map_center[1], 13)

    peak_coords = np.asarray(peak_coords);
    # gmap.scatter(peak_coords[:,0], peak_coords[:,1], '#3B0B39', size=40, marker=False)
    # gmap.scatter(peak_coords[:,0], peak_coords[:,1], '#000000', size=400, marker=True)
    gmap.heatmap(peak_coords[:,0], peak_coords[:,1], radius=20, opacity=0.5, gradient=[(50,50,50,0), (255,0,0,1), (255, 0, 0, 1)])

    gmap.draw("easy_to_read_map.html")

if __name__ == "__main__":

    peak_points, peak_coords = find_maxima()

    draw_2D_plot(mixture_distribution, peak_points)

    draw_on_google_map(BERLIN_GPS, peak_coords)

    print "System Output: "
    print "The three GPS coordinates: "
    print peak_coords

import numpy as np
import math
import gmplot
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vectors import *

SW_lat = 52.464011 #(Latitude)
SW_lon = 13.274099 #(Longitude)
EARTH_RADIUS = 6371000 #Earth radius in meter

GATE_GPS = [ 52.516288, 13.377689]


BERLIN_LAT = 52.5167
BERLIN_LON = 13.3833

#The spree river could be represented with line segments between the following coordinates
SPREE_GPS = [
    [52.529198,13.274099],
    [52.531835,13.29234],
    [52.522116,13.298541],
    [52.520569,13.317349],
    [52.524877,13.322434],
    [52.522788,13.329],
    [52.517056,13.332075],
    [52.522514,13.340743],
    [52.517239,13.356665],
    [52.523063,13.372158],
    [52.519198,13.379453],
    [52.522462,13.392328],
    [52.520921,13.399703],
    [52.515333,13.406054],
    [52.514863,13.416354],
    [52.506034,13.435923],
    [52.496473,13.461587],
    [52.487641,13.483216],
    [52.488739,13.491456],
    [52.464011,13.503386]]

SATELLITE_START_GPS = [52.590117, 13.39915]
SATELLITE_END_GPS = [52.437385, 13.553989]
def coordinate_conversion(P_lat, P_lon):
    """
    Args:
        P_lat:
        P_lon:

    Returns:
        A point in XY coordinate.

    """
    P_x = (P_lon - SW_lon) * math.cos(SW_lat * math.pi / 180) * 111.323
    P_y = (P_lat - SW_lat) * 111.323

    return P_x, P_y

def gaussian_distribution(x, mu, sig):
    return 1./(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def lognormal_distribution(x, mean, mode):
    return True


def distance_to_linesegment(pnt, start, end):
    """
    Calculate the shortest distance between a point and a line segment.
    >>> distance_to_linesegment([0, 0, 0], [1, 0, 0], [1, 1, 0])
    1.0
    >>> distance_to_linesegment([0, 0, 0], [1, 0, 0],[0, 1, 0])
    0.7071067811865476
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
    return dist


def spree_distribution(x,y, mean= 0.0, sigma = 1.0):
    SPREE_coords = []
    for GPS_coord in SPREE_GPS:
        P_x, P_y = coordinate_conversion(GPS_coord[0], GPS_coord[1])
        SPREE_coords.append([P_x, P_y, 0])
    distances = []
    idx = 0
    for idx in range(len(SPREE_coords)-1):
        start_point = SPREE_coords[idx]
        end_point = SPREE_coords[idx+1]
        distances.append(distance_to_linesegment([x,y,0], start_point, end_point))
    shortest_distance = min(distances)
    return stats.norm(mean, sigma).pdf(shortest_distance)


def gate_distribution(x, y, mean = 4.7, mode = 3.877):
    GATE_coord = coordinate_conversion(GATE_GPS[0], GATE_GPS[1])
    shortest_distance = length(vector([GATE_coord[0], GATE_coord[1],0], [x, y, 0]))
    theta = math.sqrt(2 * math.log(mean/mode)/3)
    sigma = (2 * math.log(mean) + math.log(mode))/3
    return stats.lognorm(s=theta, scale=math.exp(sigma)).pdf(shortest_distance)


def satellite_distribution(x, y, mean=0.0, sigma=1.0):
    start_coord = coordinate_conversion(SATELLITE_START_GPS[0], SATELLITE_START_GPS[1])
    end_coord = coordinate_conversion(SATELLITE_END_GPS[0], SATELLITE_END_GPS[1])

    shortest_distance = distance_to_linesegment([x,y,0], [start_coord[0], start_coord[1], 0], [end_coord[0], end_coord[1], 0])
    return stats.norm(mean, sigma).pdf(shortest_distance)


def mixture_distribution(x, y):
    return 1.0/3 * spree_distribution(x, y) + 1.0/3 * gate_distribution(x, y) + 1.0/3 * satellite_distribution(x, y)

def convert_spree(spree_coords):

    XY_coords = []
    for GPS_coord in spree_coords:
        P_x, P_y = coordinate_conversion(GPS_coord[0], GPS_coord[1])
        XY_coords.append([P_x, P_y])

    return XY_coords

def draw_2D_plot(pdf):
    # display predicted scores by the model as a contour plot
    x = np.linspace(0.0, 20.0)
    y = np.linspace(0.0, 12.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    probs = np.zeros(2500)

    idx=0
    for point in XX:
        probs[idx] = pdf(point[0], point[1])
        idx = idx+1

    probs = probs.reshape((50, 50));

    CS = plt.contour(X, Y, probs);
    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    spree_coords = convert_spree(SPREE_GPS)
    spree_coords = np.asarray(spree_coords)
    plt.plot(spree_coords[:, 0], spree_coords[:, 1], 'ro')

    GATE_coord = coordinate_conversion(GATE_GPS[0], GATE_GPS[1])
    plt.plot(GATE_coord[0], GATE_coord[1], 'gD')

    start_coord = coordinate_conversion(SATELLITE_START_GPS[0], SATELLITE_START_GPS[1])
    end_coord = coordinate_conversion(SATELLITE_END_GPS[0], SATELLITE_END_GPS[1])
    plt.plot([start_coord[0], start_coord[1]], [end_coord[0], end_coord[1]], color='y', ls='--', lw=1)

    plt.axis([-1, 20, -1, 10])

    plt.title('PDF')
    plt.axis('tight')
    plt.show()

def draw_3D_plot(pdf):
    # display predicted scores by the model as a contour plot
    x = np.linspace(0.0, 20.0, num=100)
    y = np.linspace(0.0, 12.0, num=100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    probs = np.zeros(10000)

    idx=0
    for point in XX:
        probs[idx] = pdf(point[0], point[1])
        idx = idx+1

    probs = probs.reshape((100, 100));

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, probs)
    plt.show()

if __name__ == "__main__":

    draw_2D_plot(mixture_distribution)

    # draw_3D_plot(spree_distribution)
    # draw_3D_plot(satellite_distribution)
    # draw_3D_plot(mixture_distribution)
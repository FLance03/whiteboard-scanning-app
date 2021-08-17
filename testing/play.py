import numpy as np
points = [np.array([63,40,61]),np.array([33,25,97]),np.array([33,98,91])]
point1, point2, point3 = [np.array(point) for point in points]
pq, qr = point2 - point1, point3 - point1
a, b, c = np.cross(pq, qr)
d = - (a * point1[0] + b * point1[1] + c * point1[2])
# The research based upon used the equation ax+by-z+c=0 with a 3D point (x,y,z). Thus, c forced to be -1 (For readability)
a, b, d, c = -a / c, -b / c, -d / c, -1
# Did not return c anymore
print(a, b, d)
import numpy as np
from scipy.spatial import distance


# Find shortest edge of a polygon
def get_shortest(polygon):
    shortest = distance.euclidean(polygon[-1], polygon[0])
    index_shortest = 0
    for counter in range(len(polygon)-1):
        new_distance = distance.euclidean(polygon[counter], polygon[counter+1])
        if new_distance < shortest:
            shortest = new_distance
            index_shortest = counter+1

    
    return index_shortest

# find intersection point of two lines
def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    print(s)
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (int(x/z), int(y/z))

# remove original line end points after creating intersection of lines
def remove_corner(polygon, index_shortest):
    intersect_x,intersext_y= get_intersect(polygon[index_shortest ],polygon[(index_shortest+1)%len(polygon)],polygon[index_shortest-1],polygon[index_shortest-2])
    print(index_shortest)
    print(intersect_x," ", intersext_y)
    polygon[index_shortest]=[intersect_x,intersext_y]
    polygon_short=np.delete(polygon,index_shortest-1,axis=0)
    return(polygon_short)

# sort bounding boy points in predefined (HAHAHA!) order
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]

   
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
    
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order

    ###### STRANGE br and tr mixed up
	return np.array([tl, tr, br, bl], dtype="float32")


# extend polygon to include black numbers
def extend_box(polygon):
    upper_left_x= polygon[0][0] - (polygon[1][0]-polygon[0][0])
    upper_left_y= polygon[0][1] - (polygon[1][1]-polygon[0][1])
    lower_left_x= polygon[3][0] - (polygon[2][0]-polygon[3][0])
    lower_left_y= polygon[3][1] - (polygon[2][1]-polygon[3][1])
    extended_polygon=np.array([[upper_left_x,upper_left_y],[lower_left_x,lower_left_y],polygon[2],polygon[1]])
    return extended_polygon 


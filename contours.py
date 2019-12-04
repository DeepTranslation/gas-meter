
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

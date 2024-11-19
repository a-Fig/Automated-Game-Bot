import objectFinder as ob
import time
import window_info
def detectObstacles(image):
    result = (ob.ALBION_OBSTACLE_MODEL.infer(image)[0].predictions)

    return result

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, pointA, pointB):
        self.pointA = pointA #type Point
        self.pointB = pointB

class Obstacle:
    def __init__(self, confidence, points):
        self.points = points #type array<Point>
        self.confidence = confidence
        offset = window_info.windowLocation()
        if offset[0] > 0 or offset[1] > 0:
            for p in self.points:
                p.x += offset[0]
                p.y += offset[1]

    def does_intersect(self, line):
        # Helper function to check if point q lies on segment pr
        def on_segment(p, q, r):
            return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                    q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))

        # Helper function to determine the orientation of three ordered points (p, q, r)
        # 0 -> p, q and r are collinear
        # 1 -> Clockwise
        # 2 -> Counterclockwise
        def orientation(p, q, r):
            val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
            if val == 0:
                return 0  # Collinear points
            return 1 if val > 0 else 2  # 1 if clockwise, 2 if counterclockwise

        # Main function to check if two segments intersect
        def segments_intersect(p1, q1, p2, q2):
            # Find the four orientations needed to determine if the segments intersect
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case: If orientations differ, segments intersect
            if o1 != o2 and o3 != o4:
                return True

            # Special cases: Check if points are collinear and if they lie on the segments
            if o1 == 0 and on_segment(p1, p2, q1): return True
            if o2 == 0 and on_segment(p1, q2, q1): return True
            if o3 == 0 and on_segment(p2, p1, q2): return True
            if o4 == 0 and on_segment(p2, q1, q2): return True

            return False  # Segments do not intersect

        # Loop through each consecutive pair of points in the obstacle's boundary
        for i in range(len(self.points) - 1):
            # Check if the line segment between `line.pointA` and `line.pointB`
            # intersects with the segment between `self.points[i]` and `self.points[i + 1]`
            if segments_intersect(line.pointA, line.pointB, self.points[i], self.points[i + 1]):
                return True  # An intersection is found

        return False  # No intersection found across any segments

class ManualObstacleImage():
    def __init__(self, image, timestamp):
        self.obstacleList = list()
        self.timestamp = timestamp
        for detected in detectObstacles(image):
            if detected.confidence > .5:
                self.obstacleList.append(Obstacle(detected.confidence, detected.points))

    def does_intersect(self,line):
        for obstacle in self.obstacleList:
            if obstacle.does_intersect(line):
                return True
        return False
class AutoObstacleImage():
    def __init__(self,):
        self.obstacleList = list()
        self.timestamp = time.time()
        for detected in detectObstacles(ob.screenshot()):
            if detected.confidence > .5:
                self.obstacleList.append(Obstacle(detected.confidence, detected.points))

    def does_intersect(self,line):
        for obstacle in self.obstacleList:
            if obstacle.does_intersect(line):
                return True
        return False

def constantObstacleScanning(imageStack, stack_lock, keep_scanning):
    while keep_scanning[0]:
        new_obstacle_image = AutoObstacleImage()
        with stack_lock:
            imageStack.append(new_obstacle_image)


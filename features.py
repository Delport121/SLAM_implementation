import numpy as np
import math
from fractions import Fraction
from scipy.odr import *

# Landmmarks
Landmarks = []

class featuresDetection:
    def __init__(self):
        # variables
        self.EPSILON = 4 #Distance that points may deviate from the centre line of the segment
        self.DELTA = 401 #The distance from every point in the seed-segment to its predicted position also should be less than a given threshold
        self.SNUM = 6 #Number of laser points in a seed segment
        self.PMIN = 20 # minimum number of laser points contained in an extracted line segment.
        self.GMAX = 20
        self.SEED_SEGMENTS = []
        self.LINE_SEGMENTS = []
        self.LASERPOINTS = []
        self.LINE_PARAMS = None
        self.NP = len(self.LASERPOINTS) - 1
        self.LMIN = 10 # minimum length of a line segment
        self.LR = 0 # real lenght of line segment
        self.PR = 0 # the number of laser points contained in the line segment
        self.FEATURES = []
        
    # euclidian distance from point1 to point2
    @staticmethod
    def dist_point2point(point1, point2):
        Px = (point1[0] - point2[0])**2
        PY = (point1[1] - point2[1])**2
        return math.sqrt(Px + PY)
    
    # distnace point to line written in general form
    def dist_point2line(self, params, point):
        A, B, C = params
        distance = abs(A*point[0] + B*point[1] + C) / math.sqrt(A**2 + B**2)
        return distance
    
    # extract twos point from a line equation under the slope intercepts form
    def line_2points(self, m, b):
        x = 5
        y = m*x + b
        x2 = 2000
        y2 = m*x2 + b
        return [(x, y), (x2, y2)]
    
    #general form to slope-intercept
    def lineForm_G2SI(self, A, B , C):
        m = -A/B
        b = -C/B
        return m, b
    
    #slope intercept to general form
    def lineForm_Si2G(self, m, b):
        A, B, C = -m, 1, -b
        if A < 0:
            A, B, C = -A, -B, -C
        den_a = Fraction(A).limit_denominator(1000).as_integer_ratio()[1]
        den_c = Fraction(C).limit_denominator(1000).as_integer_ratio()[1]

        gcd = np.gcd(den_a, den_c)
        lcm = den_a * den_c / gcd

        A = A * lcm
        B = B * lcm
        C = C * lcm
        return A, B, C 
    
    # The methods does not account for when the lines are parallel. It calculates the line intersection from the general form
    def line_intersect_general(self, params1, params2):
        A1, B1, C1 = params1
        A2, B2, C2 = params2
        x = (C1*B2 - C2*B1) / (A2*B1 - A1*B2)
        y = (A1*C2 - A2*C1) / (A2*B1 - A1*B2)
        return x, y
    
    # Calcultate the slope-intercept paramters from two points
    def points_2line(self, point1, point2):
        m, b = 0, 0
        if point2[0] == point1[0]:
            pass
        else:
            m = (point2[1] - point1[1]) / (point2[0] - point1[0])
            b = point2[1] - m*point2[0]
        return m, b  
    
    # 
    def projection_point2line(self, point , m, b):
        x,y = point
        m2 = -1/m
        c2 = y - m2*x
        intersection_x = -(b - c2) / (m - m2)
        intersection_y = m2*intersection_x + c2
        return intersection_x, intersection_y
    
    def AD2pos(self, distance, angle, robot_position):
        x = distance * math.cos(angle) + robot_position[0]
        y = -distance * math.sin(angle) + robot_position[1]
        return (int(x), int(y))
    
    def laser_points_set(self, data):
        self.LASERPOINTS = []
        if not data:
            pass
        else:
            for point in data:
                coordinates = self.AD2pos(point[0], point[1], point[2])
                self.LASERPOINTS.append([coordinates, point[1]])
        self.NP = len(self.LASERPOINTS) - 1

    #Define a function (quadratic in our case) to fit the data with
    def linear_func(self, p, x):
        m, b = p
        return m * x + b
    
    #Orthogonal Distance Regression (Minimise the distance between the data points and the tangent of model) (It fits a line to some of the data points)
    def odr_fit(self, laser_points): 
        x = np.array([i[0][0] for i in laser_points])
        y = np.array([i[0][1] for i in laser_points])

        #Create a model fitting
        linear_model = Model(self.linear_func)

        # Create a RealData object using our initiated data from above
        data = RealData(x, y)

        # Set up ODR with the model and data
        odr_model = ODR(data, linear_model, beta0=[0., 0.])

        #Run the regression
        out = odr_model.run()
        m, b = out.beta
        return m, b
    
    def predictPoint(self, line_params, sensed_point, robotpos):
        m, b = self.points_2line(robotpos, sensed_point)
        params1 = self.lineForm_Si2G(m, b)
        predx, predy = self.line_intersect_general(params1, line_params)
        return predx, predy
    
    def seed_segment_detection(self, robot_position, break_point_ind):
        flag = True
        self.NP = max(0, self.NP)
        self.SEED_SEGMENTS = []
        for i in range(break_point_ind, (self.NP - self.PMIN)):
            predicted_points_to_draw = []
            j = i + self.SNUM
            m, c = self.odr_fit(self.LASERPOINTS[i:j])

            params = self.lineForm_Si2G(m, c)

            for k in range (i, j):
                predicted_point = self.predictPoint(params, self.LASERPOINTS[k][0], robot_position)
                predicted_points_to_draw.append(predicted_point)
                d1 = featuresDetection.dist_point2point(predicted_point, self.LASERPOINTS[k][0])

                if d1 > self.DELTA:
                    flag = False
                    break

                d2 = self.dist_point2line(params, self.LASERPOINTS[k][0]) #Second argument is correct, video corrects this

                if d2 > self.EPSILON:
                    flag = False
                    break

            if flag:
                self.LINE_PARAMS = params
                return [ self.LASERPOINTS[i:j], predicted_points_to_draw, (i, j)] # Detected seed segment, predicted points, start and end indexes of the seed segment
        return False
    
    def seed_segment_growing(self, indices, breakpoint):
        line_eq = self.LINE_PARAMS
        i, j = indices
        #Beginning and Final points in a line segment
        PB, PF = max(breakpoint, i - 1), min(j + 1, len(self.LASERPOINTS) - 1)

        while self.dist_point2line(line_eq, self.LASERPOINTS[PF][0]) < self.EPSILON:
            if PF > self.NP - 1:
                break
            else:
                m, b = self.odr_fit(self.LASERPOINTS[PB:PF])
                line_eq = self.lineForm_Si2G(m, b)

                POINT = self.LASERPOINTS[PF][0]

            PF = PF + 1
            NEXTPOINT = self.LASERPOINTS[PF][0]
            if featuresDetection.dist_point2point(POINT, NEXTPOINT) > self.GMAX:
                break

        PF = PF - 1

        while self.dist_point2line(line_eq, self.LASERPOINTS[PB][0]) < self.EPSILON:

            if PB < breakpoint:
                break
            else:
                m, b = self.odr_fit(self.LASERPOINTS[PB:PF])
                line_eq = self.lineForm_Si2G(m, b)
                POINT = self.LASERPOINTS[PB][0]

            PB = PB - 1
            NEXTPOINT = self.LASERPOINTS[PB][0]
            if featuresDetection.dist_point2point(POINT, NEXTPOINT) > self.GMAX:
                break
                
        PB = PB + 1

        LR = featuresDetection.dist_point2point(self.LASERPOINTS[PB][0], self.LASERPOINTS[PF][0])
        PR = len(self.LASERPOINTS[PB:PF])

        if (LR >= self.LMIN) and (PR >= self.PMIN):
            self.LINE_PARAMS = line_eq
            m, b = self.lineForm_G2SI(line_eq[0], line_eq[1], line_eq[2])
            self.two_points = self.line_2points(m, b)
            self.LINE_SEGMENTS.append((self.LASERPOINTS[PB + 1][0], self.LASERPOINTS[PF - 1][0]))
            return [self.LASERPOINTS[PB:PF], self.two_points,(self.LASERPOINTS[PB + 1][0], self.LASERPOINTS[PF - 1][0]), PF, line_eq, (m, b)]
        else:
            return False
        
    def lineFeats2point(self):
        new_rep = [] # the new representation of the features 

        for feature in self.FEATURES:
            projection = self.projection_point2line((0,0), feature[0][0], feature[0][1])
            new_rep.append([feature[0], feature[1], projection])

        return new_rep
    


# def dist_point2point(point1, point2):
#         Px = (point1[0] - point2[0])**2
#         PY = (point1[1] - point2[1])**2
#         return math.sqrt(Px + PY)
    
    
def landmark_association(landmarks):
    thresh = 10
    for l in landmarks:

        flag = False 
        for i, Landmark in enumerate(Landmarks):
            print(l[2])
            print(Landmark[2])
            dist = featuresDetection.dist_point2point(l[2], Landmark[2]) #Seeks two points
            if dist < thresh:
                if not is_overlap(l[1], Landmark[1]):
                    continue
                else:
                    Landmarks.pop(i)
                    Landmarks.insert(i, l)
                    flag = True
                    break
        if not flag:
            Landmarks.append(l)

def is_overlap(seg1, seg2):
    lenght1 = featuresDetection.dist_point2point(seg1[0], seg1[1])
    lenght2 = featuresDetection.dist_point2point(seg2[0], seg2[1])
    center1 = ((seg1[0][0] + seg1[1][0]) / 2, (seg1[0][1] + seg1[1][1]) / 2)
    center2 = ((seg2[0][0] + seg2[1][0]) / 2, (seg2[0][1] + seg2[1][1]) / 2)
    dist = featuresDetection.dist_point2point(center1, center2)
    if dist > (lenght1 + lenght2) / 2:
        return False
    else:
        return True
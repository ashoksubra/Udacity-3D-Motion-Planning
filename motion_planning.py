import argparse
import time
import msgpack
import utm
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt

from planning_utils import a_star, heuristic, create_grid, prune_path_bres, collinearity_check
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local



class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def point(p):
        return np.array([p[0], p[1], 1.]).reshape(1, -1)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        # Read global_home location (lat0, lon0) values from colliders.csv 1st line

        filename = 'colliders.csv'
        latlong = []

        latlong = open(filename).readline().split()
        lat0, lon0 = latlong[1],latlong[3]
        print ('Globa_home Lattitude & Longitude', lat0, lon0)

        # TODO: set home position to (lon0, lat0, 0) (Global_home)
        self.set_home_position(lat0,lon0,0.00)

        # TODO: retrieve current global position
        #print ('self lat lon & alt', self._latitude, self._longitude, self._altitude)
        #print (' 1:',self.global_position[0],self.global_position[1],self.global_position[2])
        #print (' 2:',self.global_home[0],self.global_home[1],self.global_home[2])

        geodetic_current_position = [self._longitude, self._latitude,  self._altitude]
        geodetic_home_coordinates = [self.global_home[0],self.global_home[1],self.global_home[2]]
        #print (' GCC - GHC : ', geodetic_current_position, geodetic_home_coordinates)

        # TODO: convert to current local position using global_to_local()

        local_position = global_to_local(geodetic_current_position, geodetic_home_coordinates)
        #print ('Local Position', local_position)

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        print (data)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # Define starting point on the grid (this is just grid center)

        grid_start = (int(local_position[0]) - north_offset, int(local_position[1] - east_offset))

        #print ('Grid Start - Grid Goal', grid_start, grid_goal)

        # TODO: convert start position to current position rather than map center
        # Set goal as some arbitrary position on the grid

        #N_offset,E_offset = map(int, raw_input().split())
        N_offset = int(input ('Enter North offset :'))
        E_offset = int(input ('Enter East offset :'))
        grid_goal = (grid_start[0]+N_offset, grid_start[1]+E_offset)

        # Run A* to find a path from start to goal

        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)

        print('Local Start and Goal: ', grid_start, grid_goal)
        path, cost = a_star(grid, heuristic, grid_start, grid_goal)

        # Plot the path after A*
        """
        plt.imshow(grid, cmap='Greys', origin='lower')

        # For the purposes of the visual the east coordinate lay along
        # the x-axis and the north coordinates long the y-axis.

        plt.plot(grid_start[1], grid_start[0], 'x')
        plt.plot(grid_goal[1], grid_goal[0], 'x')

        if path is not None:
           pp = np.array(path)
           plt.plot(pp[:, 1], pp[:, 0], 'g')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.show()
        """
        # TODO: prune path to minimize number of waypoints

        ppath = prune_path_bres(grid, path)

        # Plot the path after Bresenham

        plt.imshow(grid, cmap='Greys', origin='lower')

        # For the purposes of the visual the east coordinate lay along
        # the x-axis and the north coordinates long the y-axis.

        plt.plot(grid_start[1], grid_start[0], 'x')
        plt.plot(grid_goal[1], grid_goal[0], 'x')

        if ppath is not None:
           pp = np.array(ppath)
           plt.plot(pp[:, 1], pp[:, 0], 'g')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.show()

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in ppath]

        # Set self.waypoints
        self.waypoints = waypoints
        print ('Way points :', self.waypoints)

        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

        # TODO: convert to current local position using global_to_local()

        local_position = global_to_local(geodetic_current_position, geodetic_home_coordinates)
        print ('Local Position End', local_position)


    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")

        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)
    drone.start()

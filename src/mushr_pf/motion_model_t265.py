#!/usr/bin/env python

# Copyright (c) 2021, Microsoft

from threading import Lock

import numpy as np
import rospy
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped
from nav_msgs.msg import Odometry
import utils

# Tune these Values!
KM_V_NOISE = 0.4  # Kinematic car velocity noise std dev
KM_DELTA_NOISE = 0.2  # Kinematic car delta noise std dev
KM_X_FIX_NOISE = 3e-2  # Kinematic car x position constant noise std dev
KM_Y_FIX_NOISE = 3e-2  # Kinematic car y position constant noise std dev
KM_THETA_FIX_NOISE = 1e-1  # Kinematic car theta constant noise std dev

"""
  Propagates the particles forward based the T265 odometry difference
"""


class KinematicMotionModelT265:

    """
    Initializes the kinematic motion model
      motor_state_topic: The topic containing motor state information
      servo_state_topic: The topic containing servo state information
      speed_to_erpm_offset: Offset conversion param from rpm to speed
      speed_to_erpm_gain: Gain conversion param from rpm to speed
      steering_angle_to_servo_offset: Offset conversion param from servo position to steering angle
      steering_angle_to_servo_gain: Gain conversion param from servo position to steering angle
      car_length: The length of the car
      particles: The particles to propagate forward
      state_lock: Controls access to particles
    """

    def __init__(
        self,
        t265_state_topic,
        particles,
        state_lock=None,
    ):
        self.last_t265_odom = None  # The most recent T265 odom message
        self.last_t265_stamp = None  # The time stamp from the previous T265 state msg
        self.particles = particles

        if state_lock is None:
            self.state_lock = Lock()
        else:
            self.state_lock = state_lock

        # Subscribe to the odometry from the T265 tracking camera
        self.tracking_sub = rospy.Subscriber(
            t265_state_topic, Odometry, self.t265_cb, queue_size=1
        )


    """
    Caches the most recent T265 odometry message
      msg: A nav_msgs/Odometry message
  """

    def t265_cb(self, msg):
        self.state_lock.acquire()
        if self.last_t265_odom is None:
            print("T265 callback called for first time....")
            self.last_t265_odom = msg  # Update T265 odom
            self.last_t265_stamp = msg.header.stamp
            self.state_lock.release()
            return
        else:
            # Propagate particles forward in place using delta odom
            dt = (msg.header.stamp - self.last_t265_stamp).to_sec()
            self.apply_odom_delta(self.particles, msg, self.last_t265_odom, dt)
            self.last_t265_odom = msg  # Update T265 odom
            self.last_t265_stamp = msg.header.stamp
            self.state_lock.release()


    def apply_odom_delta(self, proposal_dist, odom_curr, odom_prev, dt):
        """
        Propagates particles forward (in-place) by applying the difference btw odoms and adding
        sampled gaussian noise
        proposal_dist: The particles to propagate
        odom_curr: current position captured by the T265
        odom_prev: last position captured by the T265
        dt: time interval since the last update
        returns: nothing
        """

        # updates in X and Y are simple
        proposal_dist[:, 0] -= odom_curr.pose.pose.position.x - odom_prev.pose.pose.position.x
        proposal_dist[:, 1] -= odom_curr.pose.pose.position.y - odom_prev.pose.pose.position.y

        # update in theta requires odom angle diff in XY plane
        proposal_dist[:, 2] -= utils.quaternion_to_angle(odom_curr.pose.pose.orientation)-utils.quaternion_to_angle(odom_prev.pose.pose.orientation)

        # delta_theta = utils.quaternion_to_angle(odom_curr.pose.pose.orientation)-utils.quaternion_to_angle(odom_prev.pose.pose.orientation)
        # delta_odom_x = odom_curr.pose.pose.position.x - odom_prev.pose.pose.position.x
        # delta_odom_y = odom_curr.pose.pose.position.y - odom_prev.pose.pose.position.y
        # print("Delta_x = " + str(delta_odom_x))
        # print("Delta_y = " + str(delta_odom_y))
        # print("Delta_theta = " + str(delta_theta))

        # Add noise
        proposal_dist[:, 0] = np.random.normal(
            loc=proposal_dist[:, 0],
            scale=KM_X_FIX_NOISE,
            size=proposal_dist[:, 0].shape,
        )
        proposal_dist[:, 1] = np.random.normal(
            loc=proposal_dist[:, 1],
            scale=KM_Y_FIX_NOISE,
            size=proposal_dist[:, 1].shape,
        )
        proposal_dist[:, 2] = np.random.normal(
            loc=proposal_dist[:, 2], scale=KM_THETA_FIX_NOISE, size=proposal_dist[:, 2].shape
        )

        # Limit particle rotation to be between -pi and pi
        proposal_dist[proposal_dist[:, 2] < -1 * np.pi, 2] += 2 * np.pi
        proposal_dist[proposal_dist[:, 2] > np.pi, 2] -= 2 * np.pi


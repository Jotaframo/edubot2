import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from python_controllers.Forward_Kinematics import forward_kinematics
import csv
import os

class EEPlotter(Node):
    def __init__(self):
        super().__init__('ee_plotter')
        self.qnames = ["Shoulder_Rotation","Shoulder_Pitch","Elbow","Wrist_Pitch"]
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb, qos)
        self.csv_path = os.path.join(os.getcwd(), 'ee_path.csv')
        with open(self.csv_path,'w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['t','x','y','z'])

    def cb(self,msg):
        ids = {n:i for i,n in enumerate(msg.name)}
        q = [msg.position[ids[n]] for n in self.qnames]
        try:
            x, y, z = forward_kinematics(q[0], q[1], q[2], q[3])
        except Exception as e:
            self.get_logger().error(f"forward_kinematics failed: {e}")
            return
        t = self.get_clock().now().seconds_nanoseconds()
        with open(self.csv_path,'a',newline='') as f:
            csv.writer(f).writerow([t[0]+t[1]*1e-9, x, y, z])
        self.get_logger().info(f"ee {x:.3f},{y:.3f},{z:.3f}")

def main():
    rclpy.init()
    n = EEPlotter()
    rclpy.spin(n)
    n.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()



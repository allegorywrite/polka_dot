import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

class SimulationManager():
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        
    def create_field(self):
        self.vis.create_window()

    def render_field(self):
        # self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def viz_run(self):
        self.vis.run()

    def destroy_field(self):
        self.vis.destroy_window()

    def create_quadcopter(self, color=[0, 0, 1]):
        # quadcpoter body
        # body = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)
        # body.paint_uniform_color([1, 0, 0])
        # quadcopter propeller
        # drone_radius = 0.005
        # propeller_radius = 0.003
        # arm_thickness = 0.0004
        drone_radius = 0.05
        propeller_radius = 0.03
        arm_thickness = 0.001
        propeller_thickness = 0.001
        propeller_color = color
        body_color = [color[0]/2, color[1]/2, color[2]/2]
        propeller0 = o3d.geometry.TriangleMesh.create_cylinder(radius=propeller_radius, height=propeller_thickness)
        propeller0.paint_uniform_color(propeller_color)
        propeller0.translate(np.array([0, drone_radius, arm_thickness]))
        propeller1 = o3d.geometry.TriangleMesh.create_cylinder(radius=propeller_radius, height=propeller_thickness)
        propeller1.paint_uniform_color(propeller_color)
        propeller1.translate(np.array([0, -drone_radius, arm_thickness]))
        propeller2 = o3d.geometry.TriangleMesh.create_cylinder(radius=propeller_radius, height=propeller_thickness)
        propeller2.paint_uniform_color(propeller_color)
        propeller2.translate(np.array([drone_radius, 0, arm_thickness]))
        propeller3 = o3d.geometry.TriangleMesh.create_cylinder(radius=propeller_radius, height=propeller_thickness)
        propeller3.paint_uniform_color(propeller_color)
        propeller3.translate(np.array([-drone_radius, 0, arm_thickness]))
        # quadcopter arm
        arm0 = o3d.geometry.TriangleMesh.create_box(width=drone_radius*2, height=arm_thickness, depth=arm_thickness)
        arm0.paint_uniform_color(body_color)
        arm0.translate(np.array([-drone_radius, 0, 0]))
        arm1 = o3d.geometry.TriangleMesh.create_box(width=arm_thickness, height=drone_radius*2, depth=arm_thickness)
        arm1.paint_uniform_color(body_color)
        arm1.translate(np.array([0, -drone_radius, 0]))
        # quadcopter
        quadcopter = o3d.geometry.TriangleMesh()
        # quadcopter += body
        quadcopter += propeller0
        quadcopter += propeller1
        quadcopter += propeller2
        quadcopter += propeller3
        quadcopter += arm0
        quadcopter += arm1
        # quadcopter.translate(np.array([0, 0, -0.05]))
        # quadcopter.rotate(0, 0, 0)
        return quadcopter
    
    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        rot_mat = np.dot(
            Rotation.from_euler('z', yaw).as_matrix(),
            np.dot(
                Rotation.from_euler('y', pitch).as_matrix(),
                Rotation.from_euler('x', roll).as_matrix()
            )
        )
        return rot_mat

    def add_quadcopter(self, x, y, z, eular=None, R=None, color=[0, 0, 1]):
        mesh = self.create_quadcopter(color=color)
        mesh.translate(np.array([x, y, z]))
        if R is None:
            R = Rotation.from_euler('xyz', eular).as_matrix()
        mesh.rotate(R)
        self.vis.add_geometry(mesh)

    def add_origin(self):
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(origin)

    def draw_trajectory(self, x, y, z, R_flat=None, eular=None, color=[0, 0, 1]):
        for i in range(len(x)):
            if eular is not None:
                self.add_quadcopter(x[i], y[i], z[i], eular=eular[i], color=color)
            else:
                R = R_flat[i].reshape(3, 3)
                self.add_quadcopter(x[i], y[i], z[i], R=R, color=color)

    def add_point(self, x, y, z):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[x, y, z]]))
        self.vis.add_geometry(pcd)

if __name__ == "__main__":
    sim_manager = SimulationManager()
    sim_manager.create_field()
    sim_manager.add_origin()
    sim_manager.add_quadcopter(0, 0, 0, 0, 0, 0)
    sim_manager.add_quadcopter(0, 0, 0.1, 0, 0, 0)
    sim_manager.viz_run()
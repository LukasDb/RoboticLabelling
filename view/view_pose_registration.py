import tkinter as tk
from tkinter import ttk
from model.scene import Scene
from control.pose_registrator import PoseRegistrator
from scipy.spatial.transform import Rotation as R


class ViewPoseRegistration(ttk.Frame):
    def __init__(self, parent, scene: Scene, registrator: PoseRegistrator) -> None:
        super().__init__(parent)
        self.scene = scene
        self.registrator = registrator

        self.title = ttk.Label(self, text="2. Pose Registration")
        self.title.grid()

        self.object_selection = ttk.Combobox(
            self, values=[o.name for o in self.scene.objects.values()]
        )

        self.object_selection.grid()

        self.position = tk.Label(self)
        self.position.grid()

        self.orientation = tk.Label(self)
        self.orientation.grid()

        # add button
        self.update_button = ttk.Button(
            self, text="Optimize", command=lambda: registrator.optimize_pose()
        )
        self.update_button.grid()

        # self.update_button = ttk.Button(
        #     self, text="Update", command=registrator.update_pose
        # )
        # self.update_button.grid()

        # self.reset_button = ttk.Button(
        #     self, text="Reset", command=registrator.reset_pose
        # )

        # st.title("2. Pose Registration")
        # c1, c2 = st.columns(2)

        # selected_camera = c2.selectbox("Select Camera", ["Camera 1", "Camera 2"])
        # selected_object = c2.selectbox("Select Object", ["Object 1", "Object 2"])
        # c2.divider()

        # registrator = PoseRegistrator()

        # pos = registrator.get_position()
        # orn = registrator.get_orientation().as_euler("xyz", degrees=True)
        # st.session_state.reg_x = pos[0]
        # st.session_state.reg_y = pos[1]
        # st.session_state.reg_z = pos[2]
        # st.session_state.reg_φ = orn[0]
        # st.session_state.reg_θ = orn[1]
        # st.session_state.reg_ψ = orn[2]

        # t_rot = c2.columns(2)
        # update_pos = lambda: registrator.set_position(
        #     [st.session_state.reg_x, st.session_state.reg_y, st.session_state.reg_z]
        # )
        # update_rot = lambda: registrator.set_orientation(
        #     R.from_euler(
        #         "xyz",
        #         [
        #             st.session_state.reg_φ,
        #             st.session_state.reg_θ,
        #             st.session_state.reg_ψ,
        #         ],
        #         degrees=True,
        #     )
        # )
        # t_rot[0].number_input("X", key="reg_x", on_change=update_pos, format="%.3f")
        # t_rot[0].number_input("Y", key="reg_y", on_change=update_pos, format="%.3f")
        # t_rot[0].number_input("Z", key="reg_z", on_change=update_pos, format="%.3f")
        # t_rot[1].number_input(
        #     "φ", key="reg_φ", on_change=update_rot, step=5.0, format="%.1f"
        # )
        # t_rot[1].number_input(
        #     "θ", key="reg_θ", on_change=update_rot, step=5.0, format="%.1f"
        # )
        # t_rot[1].number_input(
        #     "ψ", key="reg_ψ", on_change=update_rot, step=5.0, format="%.1f"
        # )

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

        self.camera_selection = ttk.Combobox(
            self, values=[c.name for c in self.scene.cameras.values()]
        )
        self.camera_selection.grid()

        self.object_selection = ttk.Combobox(
            self, values=[o.name for o in self.scene.objects]
        )
        self.object_selection.grid()

        self.position = tk.Label(self)
        self.position.grid()

        self.orientation = tk.Label(self)
        self.orientation.grid()

        # self.update_button = ttk.Button(
        #     self, text="Update", command=registrator.update_pose
        # )
        # self.update_button.grid()

        # self.reset_button = ttk.Button(
        #     self, text="Reset", command=registrator.reset_pose
        # )

        self.live_preview = tk.Label(self)
        self.live_preview.grid()

        self.selected_image = tk.Label(self)
        self.selected_image.grid()

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

        # c2.divider
        # pose_controls = c2.columns(2)
        # if pose_controls[0].button("Optimize Pose"):
        #     registrator.optimize_pose()
        # if pose_controls[1].button("Reset Pose"):
        #     registrator.reset()
        #     st.experimental_rerun()

        # c2.divider()
        # img_control = c2.columns(2)
        # if img_control[0].button("Capture Image"):
        #     registrator.capture_image()

        # if img_control[1].button("Delete Captured Images"):
        #     registrator.clear_cache()

        # sel_img_index = c2.radio(
        #     "Captured Images",
        #     list(range(len(registrator._captured_images))),
        #     format_func=lambda x: f"Image {x:2}",
        # )

        # live = registrator.get_live_img()
        # sel_img = registrator.get_selected_img(sel_img_index)

        # c1.image(
        #     live,
        #     caption="Live preview",
        # )
        # if sel_img is not None:
        #     c1.image(
        #         sel_img,
        #         caption="Selected Image",
        #     )

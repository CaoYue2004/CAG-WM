from typing import Optional
from math import cos, sin
import numpy as np
import torch
import dnnlib
from . import legacy
import cv2
import csv
from typing import Optional, Tuple
import importlib

from .visualisation import Visualisation
from ..intervention import Intervention
from ..interimtarget import InterimTarget, InterimTargetDummy
from training.triplane_original import TriPlaneGenerator
from torch_utils import misc
from .camera_utils import LookAtPoseSampler
from ..util.coordtransform import tracking3d_to_vessel_cs


def _diff_tensors(src_module, dst_module):
    def named_params_and_buffers(m):
        out = {}
        for n, p in m.named_parameters(recurse=True):
            out[n] = p
        for n, b in m.named_buffers(recurse=True):
            out[n] = b
        return out

    src = named_params_and_buffers(src_module)
    dst = named_params_and_buffers(dst_module)

    missing = []      
    mismatched = []   
    extra = []        

    for n, t in dst.items():
        if n not in src:
            missing.append(n)
        else:
            if tuple(src[n].shape) != tuple(t.shape):
                mismatched.append((n, tuple(src[n].shape), tuple(t.shape)))
    for n in src.keys():
        if n not in dst:
            extra.append(n)

    print(f"[DIFF] missing in src -> {len(missing)}")
    for n in missing[:50]:
        print("  -", n)
    if len(missing) > 50:
        print("  ...")

    print(f"[DIFF] shape mismatched -> {len(mismatched)}")
    for n, s1, s2 in mismatched[:50]:
        print(f"  - {n}: src{ s1 } vs dst{ s2 }")
    if len(mismatched) > 50:
        print("  ...")

    print(f"[DIFF] extra in src -> {len(extra)}")
    for n in extra[:50]:
        print("  -", n)
    if len(extra) > 50:
        print("  ...")

def camera_mat(a, b, r):
    r = float(r) / 157.7
    theta = np.deg2rad(float(b) - 90)  
    phi = np.deg2rad(float(a))  

    x_camera = r * np.sin(theta) * np.cos(phi)
    y_camera = r * np.sin(theta) * np.sin(phi)
    z_camera = r * np.cos(theta)

    z_xc, z_yc, z_zc = -x_camera, -y_camera, -z_camera
    x_xc, x_yc, x_zc = -1 / (x_camera + 10e-6), 1 / (y_camera + 10e-6), 0
    y_xc, y_yc, y_zc = ((z_yc * x_zc - x_yc * z_zc), -(z_xc * x_zc - z_zc * x_xc), (z_xc * x_yc - z_yc * x_xc))
    if y_zc > 0:
        x_xc, x_yc, x_zc = -x_xc, -x_yc, -x_zc
        y_xc, y_yc, y_zc = -y_xc, -y_yc, -y_zc

    D = np.array([[x_xc, y_xc, z_xc],
                  [x_yc, y_yc, z_yc],
                  [x_zc, y_zc, z_zc]])

    D_prime = D / (np.linalg.norm(D, axis=0) + 10e-6)

    R = D_prime
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x_camera, y_camera, z_camera]
    return T, [x_camera, y_camera, z_camera, 1]


def grid_to_world_batch(grid_xyz: np.ndarray,
                        bbox_min=-0.5,
                        bbox_max=0.5,
                        grid_size=512,
                        center=True,
                        clip=True) -> np.ndarray:
    grid_xyz = np.asarray(grid_xyz, dtype=np.float32)
    bbox_min = np.asarray(bbox_min, dtype=np.float32).reshape(1, 3) if np.ndim(bbox_min) == 1 else np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32).reshape(1, 3) if np.ndim(bbox_max) == 1 else np.asarray(bbox_max, dtype=np.float32)

    single = (grid_xyz.ndim == 1)
    if single:
        grid_xyz = grid_xyz.reshape(1, 3)

    if clip:
        grid_xyz = np.clip(grid_xyz, 0.0, float(grid_size - 1))

    if center:
        grid_xyz = grid_xyz + 0.5  

    norm = grid_xyz / float(grid_size - 1)
    world = bbox_min + norm * (bbox_max - bbox_min)

    return world[0] if single else world


class CAGRender(Visualisation):
    def __init__(
        self,
        network_pkl: str,
        intervention: Intervention,
        ppa: float,
        psa: float,
        reload_modules: bool,
        seed: int,
        display_size: Tuple[float, float] = (512, 512),
        color: Tuple[float, float, float, float] = (0, 0, 0, 0),
        interim_target: Optional[InterimTarget] = None,
        angles_csv: Optional[str] = None,
        device: str = "cuda:0",
    ) -> None:
        self.network_pkl = network_pkl
        self.intervention = intervention
        self.ppa = float(ppa)
        self.psa = float(psa)
        self.reload_modules = bool(reload_modules)
        self.seed = int(seed)
        self.display_size = display_size
        self.color = color

        simulation = self.intervention.simulation
        simulation.init_visual_nodes = True
        simulation.display_size = display_size
        simulation.target_size = self.intervention.target.threshold

        self.initial_orientation = None
        self._initialized = False
        self._theta_x = intervention.fluoroscopy.image_rot_zx[1] * np.pi / 180
        self._theta_z = intervention.fluoroscopy.image_rot_zx[0] * np.pi / 180
        self._initial_direction = None
        self._initial_look_at = None
        self._distance = None
        self._sofa = None
        self._sofa_gl = None
        self._opengl_gl = None
        self._opengl_glu = None
        self._pygame = None

        self.interim_target = interim_target or InterimTargetDummy()
        self.interim_targets = []
        simulation.interim_target_size = self.interim_target.threshold

        self.angles_csv = angles_csv
        self.frame_idx = 0  

        self._angles = None
        if self.angles_csv is not None:
            rows = []
            with open(self.angles_csv, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append((float(r["ppa"]), float(r["psa"])))
            self._angles = rows  # list[(ppa, psa)]

        simulation = self.intervention.simulation
        self.target = np.zeros((3,), dtype=np.float32)

        simulation.target_size = self.intervention.target.threshold
        simulation.interim_target_size = self.interim_target.threshold

        self.episode_nr = 0
        self.reached = False                 
        self._traj_2d = []                   
        self.background = None              

        self.device = torch.device(device)
        self.G = None

    def render(self) -> None:
        self._set_angles_by_frame(self.frame_idx)
        self.frame_idx += 1
        simulation = self.intervention.simulation
        if self.interim_target.reached:
            it_to_remove = self.interim_targets.pop(0)
            simulation.remove_interim_target(it_to_remove)

        focal_length = 7.191
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.device)
        z = torch.from_numpy(np.random.RandomState(self.seed).randn(1, self.G.z_dim)).to(self.device)

        cam2world_pose = LookAtPoseSampler.sampleDSA(self.ppa, self.psa, 829.4015277, device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sampleDSA(self.ppa, self.psa, 829.4015277, device=self.device)

        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = self.G.mapping(z, conditioning_params, truncation_psi=0.7, truncation_cutoff=14)
        img_dict, all_depth, all_density = self.G.synthesis(ws, camera_params)
        img = img_dict['image'][0]

        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = np.array(img.cpu().permute(1, 2, 0))

        img_np = img_np.astype(np.uint8)
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)
        elif img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)

        angio = np.ascontiguousarray(img_np)

        img_left = angio.copy()
        pos = self.intervention.simulation.dof_positions
        pos_world = grid_to_world_batch(pos, bbox_min=-0.5, bbox_max=0.5, grid_size=512, center=True, clip=True)
        T, world_s = camera_mat(self.ppa, self.psa, 829.4015277)
        K = np.array([[7.191, 0, 0.5], [0, 7.191, 0.5], [0, 0, 1]])
        prev_uv = None
        pixel_points = []

        for i in range(pos_world.shape[0]):
            x, y, z = pos_world[i].tolist()
            P_world = np.array([x, y, z, 1.0], dtype=np.float64)
            P_cam = np.dot(np.linalg.inv(T), P_world)

            P = np.dot(K, P_cam[:3])
            P_pix = P / P[2] * 512

            u = int(round(P_pix[0]))
            v = int(round(P_pix[1]))

            if 0 <= u < 512 and 0 <= v < 512:
                uv = (u, v)
                if uv != prev_uv:  
                    pixel_points.append(uv)
                    prev_uv = uv

        if len(pixel_points) >= 2:
            pts = np.array(pixel_points, dtype=np.int32)
            cv2.polylines(img_left, [pts], False, (0, 255, 0), 2)

        target = self.target
        target_world = grid_to_world_batch(target, bbox_min=-0.5, bbox_max=0.5, grid_size=512, center=True, clip=True)
        t_x, t_y, t_z = target_world.tolist()
        t_world = np.array([t_x, t_y, t_z, 1.0], dtype=np.float64)
        t_cam = np.dot(np.linalg.inv(T), t_world)
        t = np.dot(K, t_cam[:3])
        t_pix = t / t[2] * 512

        u_t = int(round(t_pix[0]))
        v_t = int(round(t_pix[1]))
        cv2.circle(img_left, (u_t, v_t), 2, (255, 0, 0), -1)

        self._pygame.event.get()
        simulation = self.intervention.simulation
        if self.interim_target.reached:
            it_to_remove = self.interim_targets.pop(0)
            simulation.remove_interim_target(it_to_remove)

        self._sofa.Simulation.updateVisual(simulation.root)
        gl = self._opengl_gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(*self.color)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        camera = simulation.camera
        width = camera.widthViewport.value
        height = camera.heightViewport.value
        print(f'width={width}, height={height}, FOV={camera.fieldOfView.value}')
        self._opengl_glu.gluPerspective(
            camera.fieldOfView.value,
            (width / height),
            camera.zNear.value,
            camera.zFar.value,
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # camera_mvm = camera.getOpenGLModelViewMatrix()
        look_at = [256, 256, 256]
        distance = 829.4015277 / 157.7 * 512
        theta = np.deg2rad(float(self.psa) - 90)  
        phi = np.deg2rad(float(self.ppa))  
        x_camera = distance * np.sin(theta) * np.cos(phi)
        y_camera = distance * np.sin(theta) * np.sin(phi)
        z_camera = distance * np.cos(theta)
        position = look_at + np.array([x_camera, y_camera, z_camera])
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        f = (look_at - position)
        f = f / np.linalg.norm(f)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        M = np.eye(4)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        M[0, 3] = -np.dot(s, position)
        M[1, 3] = -np.dot(u, position)
        M[2, 3] = np.dot(f, position)

        camera_mvm = M.T.reshape(-1)

        gl.glMultMatrixd(camera_mvm)
        self._sofa_gl.draw(simulation.root)
        gl = self._opengl_gl
        height = camera.heightViewport.value
        width = camera.widthViewport.value

        buffer = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image_array = np.fromstring(buffer, np.uint8)

        if image_array.shape:
            sofa_img = image_array.reshape(height, width, 3)
            # OpenGL → image coords
            sofa_img = np.flipud(sofa_img)  
            sofa_img = np.rot90(sofa_img, k=1)  
        else:
            sofa_img = np.zeros((height, width, 3))
        self._pygame.display.flip()
        
        img_left_512 = cv2.resize(img_left, (512, 512), interpolation=cv2.INTER_AREA)
        sofa_512 = cv2.resize(sofa_img, (512, 512), interpolation=cv2.INTER_AREA)

        out = np.concatenate([img_left_512, sofa_512], axis=1)  # (512, 1024, 3)
        return out

    def reset(self, episode_nr: int = 0) -> None:
        self.frame_idx = 0
        self.episode_nr = episode_nr
        self.reached = False  
        self._traj_2d = []  

        fluoroscopy = self.intervention.fluoroscopy
        self.target = self.intervention.target.coordinates3d
        self.target = tracking3d_to_vessel_cs(
            self.target,
            fluoroscopy.image_rot_zx,
            fluoroscopy.image_center
        )
        print("[CAGRender] before open_url", flush=True)
        with dnnlib.util.open_url(self.network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)  # type: ignore

        # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
        if self.reload_modules:
            print("Reloading Modules!")
            G_new = TriPlaneGenerator(*self.G.init_args, **self.G.init_kwargs).eval().requires_grad_(False).to(self.device)
            _diff_tensors(self.G, G_new)
            misc.copy_params_and_buffers(self.G, G_new, require_all=True)
            G_new.neural_rendering_resolution = self.G.neural_rendering_resolution
            G_new.rendering_kwargs = self.G.rendering_kwargs
            self.G = G_new

        focal_length = 7.191
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.device)
        z = torch.from_numpy(np.random.RandomState(self.seed).randn(1, self.G.z_dim)).to(self.device)

        cam2world_pose = LookAtPoseSampler.sampleDSA(self.ppa, self.psa, 829.4015277, device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sampleDSA(self.ppa, self.psa, 829.4015277, device=self.device)

        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = self.G.mapping(z, conditioning_params, truncation_psi=0.7, truncation_cutoff=14)
        img_dict, all_depth, all_density = self.G.synthesis(ws, camera_params)
        img = img_dict['image'][0]

        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = np.array(img.cpu().permute(1, 2, 0))

        img_np = img_np.astype(np.uint8)
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)
        elif img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)

        self.background = np.ascontiguousarray(img_np)

        simulation = self.intervention.simulation
        # pylint: disable=no-member
        self._sofa = self._sofa or importlib.import_module("Sofa")
        self._sofa_gl = self._sofa_gl or importlib.import_module("Sofa.SofaGL")
        self._opengl_gl = self._opengl_gl or importlib.import_module("OpenGL.GL")
        self._opengl_glu = self._opengl_glu or importlib.import_module("OpenGL.GLU")
        self._pygame = self._pygame or importlib.import_module("pygame")
        if not self._initialized:
            self._pygame.display.init()
            flags = (
                    self._pygame.DOUBLEBUF | self._pygame.OPENGL | self._pygame.RESIZABLE
            )
            self._pygame.display.set_mode(self.display_size, flags)
            self._initialized = True

        gl = self._opengl_gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(*self.color)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        self._sofa.SofaGL.glewInit()
        self._sofa.Simulation.initVisual(simulation.root)
        self._sofa.Simulation.initTextures(simulation.root)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        camera = simulation.camera
        width = camera.widthViewport.value
        height = camera.heightViewport.value
        self._opengl_glu.gluPerspective(
            camera.fieldOfView.value,
            (width / height),
            camera.zNear.value,
            camera.zFar.value,
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        if self.initial_orientation is not None:
            self._theta_x = 0.0
            self._theta_z = 0.0
            self.rotate(0, 0)
        self.initial_orientation = np.array(
            [
                camera.orientation[3],
                camera.orientation[0],
                camera.orientation[1],
                camera.orientation[2],
            ]
        )
        position = camera.position
        look_at = camera.lookAt
        self._initial_direction = position - look_at
        self._distance = np.linalg.norm(self._initial_direction)
        self._initial_direction = self._initial_direction / self._distance
        fluoroscopy = self.intervention.fluoroscopy

        vessel_tree = self.intervention.vessel_tree
        vessel_low = vessel_tree.coordinate_space.low
        vessel_high = vessel_tree.coordinate_space.high
        vessel_center = (vessel_high + vessel_low) / 2

        if fluoroscopy.image_center != [0, 0, 0]:
            look_at[0] = fluoroscopy.image_center[0]
            look_at[1] = fluoroscopy.image_center[1]
            look_at[2] = fluoroscopy.image_center[2]
        else:
            look_at[0] = vessel_center[0]
            look_at[1] = vessel_center[1]
            look_at[2] = vessel_center[2]

        self._initial_look_at = look_at
        self._theta_x = fluoroscopy.image_rot_zx[1] * np.pi / 180
        self._theta_z = fluoroscopy.image_rot_zx[0] * np.pi / 180
        camera.lookAt = self._initial_look_at
        self.rotate(0, 0)
        target = self.intervention.target.coordinates3d
        target = tracking3d_to_vessel_cs(
            target, fluoroscopy.image_rot_zx, fluoroscopy.image_center
        )
        simulation.target_node.MechanicalObject.translation = [
            target[0],
            target[1],
            target[2],
        ]
        self._sofa.Simulation.init(simulation.target_node)

        fluoroscopy = self.intervention.fluoroscopy

        if self.interim_target.all_coordinates3d:
            interim_targets_vcs = tracking3d_to_vessel_cs(
                self.interim_target.all_coordinates3d,
                fluoroscopy.image_rot_zx,
                fluoroscopy.image_center,
            )
            self.interim_targets = self.intervention.simulation.add_interim_targets(
                list(interim_targets_vcs)
            )

    def close(self) -> None:
        pass

    def _set_angles_by_frame(self, t: int):
        if not self._angles:
            return
        idx = int(t) % len(self._angles)
        self.ppa, self.psa = self._angles[idx]

    def _read_sofa_frame(self) -> np.ndarray:
        gl = self._opengl_gl
        camera = self.intervention.simulation.camera
        w = int(camera.widthViewport.value)
        h = int(camera.heightViewport.value)

        gl.glReadBuffer(gl.GL_BACK)
        buf = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        img = arr.reshape(h, w, 3)
        img = np.flipud(img)  
        return img.copy()

    def _fov_y_from_f(self, f_norm: float, H: int) -> float:
        fy_px = float(f_norm) * float(H)
        fov_y = 2.0 * np.arctan((H / 2.0) / (fy_px + 1e-9))
        return float(fov_y * 180.0 / np.pi)

    def _set_opengl_matrices_from_dsa(self, gl, glu, width: int, height: int,
                                      primary_angle: float, secondary_angle: float,
                                      dsp: float, f_norm: float,
                                      z_near: float, z_far: float,
                                      fix_axes: bool = True) -> None:
        # 1) Projection
        fov_y = self._fov_y_from_f(f_norm, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(fov_y, float(width) / float(height), z_near, z_far)

        # 2) ModelView = world2cam
        # T = cam2world (4x4)
        T, _ = camera_mat(primary_angle, secondary_angle, dsp)
        world2cam = np.linalg.inv(T)

        if fix_axes:
            C = np.diag([1.0, -1.0, -1.0, 1.0])  
            world2cam = C @ world2cam

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glMultMatrixd(world2cam.T.astype(np.float64))

    def _crop_half_and_concat(self, angio: np.ndarray, sofa: np.ndarray,
                              out_h: int = 512, half_w: int = 256) -> np.ndarray:

        if angio.ndim == 2:
            angio = np.repeat(angio[..., None], 3, axis=2)
        if sofa.ndim == 2:
            sofa = np.repeat(sofa[..., None], 3, axis=2)
        if angio.shape[2] == 1:
            angio = np.repeat(angio, 3, axis=2)
        if sofa.shape[2] == 1:
            sofa = np.repeat(sofa, 3, axis=2)

        angio_512 = cv2.resize(angio, (out_h, out_h), interpolation=cv2.INTER_AREA)
        sofa_512 = cv2.resize(sofa, (out_h, out_h), interpolation=cv2.INTER_AREA)

        angio_half = angio_512[:, :half_w, :]  # (512, 256, 3)
        sofa_half = sofa_512[:, -half_w:, :]  # (512, 256, 3)

        out = np.concatenate([angio_half, sofa_half], axis=1)  # (512, 512, 3)
        return out

    def translate(self, velocity: np.array):
        simulation = self.intervention.simulation
        dt = simulation.root.dt.value
        camera = simulation.camera

        position = camera.position
        position += velocity * dt
        camera.position = position

        look_at = camera.lookAt
        look_at += velocity * dt
        camera.lookAt = look_at

    def zoom(self, velocity: float):
        simulation = self.intervention.simulation
        dt = simulation.root.dt.value
        camera = simulation.camera

        position = camera.position
        look_at = camera.lookAt
        direction = look_at - position
        direction = (
            direction
            / (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
        )
        position += direction * velocity * dt
        self._distance -= velocity * dt
        camera.position = position

    def rotate(self, lao_rao_speed: float, cra_cau_speed: float):
        simulation = self.intervention.simulation
        dt = simulation.root.dt.value
        camera = simulation.camera

        look_at = camera.lookAt
        self._theta_x += cra_cau_speed * dt
        self._theta_z += lao_rao_speed * dt
        theta_x = self._theta_x
        theta_z = self._theta_z

        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, cos(theta_x), -sin(theta_x)],
                [0, sin(theta_x), cos(theta_x)],
            ]
        )
        rotation_z = np.array(
            [
                [cos(theta_z), -sin(theta_z), 0],
                [sin(theta_z), cos(theta_z), 0],
                [0, 0, 1],
            ]
        )
        rotation = np.matmul(rotation_z, rotation_x)
        offset = np.matmul(rotation, self._initial_direction * self._distance)

        camera.position = look_at + np.array(offset)

        camera_rot_x = np.array([cos(theta_x / 2), sin(theta_x / 2), 0, 0])
        camera_rot_z = np.array([cos(theta_z / 2), 0, 0, sin(theta_z / 2)])

        camera_orientation = self._quat_mult(camera_rot_x, self.initial_orientation)
        camera_orientation = self._quat_mult(camera_rot_z, camera_orientation)

        camera.orientation = np.array(
            [
                camera_orientation[1],
                camera_orientation[2],
                camera_orientation[3],
                camera_orientation[0],
            ]
        )

    @staticmethod
    def _quat_mult(x, y):
        return np.array(
            [
                x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
                x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2],
                x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1],
                x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0],
            ]
        )

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + eps)

def gluLookAt_modelview(eye, center, up):
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = _normalize(center - eye)  # forward
    s = _normalize(np.cross(f, up))  # right
    u = np.cross(s, f)  # true up

    M = np.eye(4, dtype=np.float64)
    M[0, 0:3] = s
    M[1, 0:3] = u
    M[2, 0:3] = -f

    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)  

    mvm_16 = M.T.reshape(-1).tolist()
    return M, mvm_16





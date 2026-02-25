from copy import deepcopy
import importlib
import math
import os
from typing import List, Optional, Tuple
import logging
import numpy as np
from .simulation import Simulation
from ..device import Device
import time


# ======================================================================
# SofaBeamAdapter
# 基于 SOFA + BeamAdapter 的介入器械物理仿真实现
# ======================================================================
class SofaBeamAdapter(Simulation):
    # ------------------------------------------------------------------
    # 初始化仿真对象
    # ------------------------------------------------------------------
    def __init__(
        self,
        friction: float = 0.1,      # 摩擦系数（用于接触约束）
        dt_simulation: float = 0.002,       # SOFA 仿真时间步长
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.friction = friction
        self.dt_simulation = dt_simulation

        self.root = None        # SOFA 根节点
        self.camera = None      # 相机节点
        self.target_node = None     # 主目标节点
        self.interim_target_node = None     # 中间目标父节点
        self.interim_targets = []       # 中间目标列表
        self.simulation_error = False   # 仿真是否发生错误（NaN 等）

        self.init_visual_nodes = False  # 是否初始化可视化节点
        self.display_size = (1, 1)      # 渲染窗口尺寸
        self.target_size = 1            # 主目标大小
        self.interim_target_size = 1    # 中间目标大小

        self._vessel_object = None      # 血管 SOFA 节点
        self._instruments_combined = None       # 合并后的器械节点

        self._sofa = None       # Sofa Python 模块
        self._sofa_runtime = None       # SofaRuntime 模块

        self._insertion_point = np.empty(())        # 器械插入点
        self._insertion_direction = np.empty(())    # 器械插入方向
        self._mesh_path: str = None                 # 血管 mesh 路径
        self._reset_add_visual: bool = None         # reset 时是否添加可视化
        self._display_size = None                   # 显示尺寸缓存
        self._coords_high = np.empty(())            # 坐标上界（用于相机）
        self._coords_low = np.empty(())             # 坐标下界（用于相机）
        self._target_size = None                    # 目标大小缓存
        self._vessel_visual_path: str = None        # 血管可视化 mesh 路径
        self._rng = np.random.default_rng()         # 随机数生成器
        self._dof_positions = None                  # 自由度（节点）3D 位置
        self._inserted_lengths = None               # 插入长度（xtip）
        self._rotations = None                      # 器械旋转角
        self._log_capture_installed = False
        self._lcp_no_convergence_seen = False
        self._stderr_cap = None
        self._stdout_cap = None

    # ------------------------------------------------------------------
    # 当前自由度位置（只读）
    # ------------------------------------------------------------------
    @property
    def dof_positions(self) -> np.ndarray:
        return self._dof_positions

    # ------------------------------------------------------------------
    # 当前插入长度（只读）
    # ------------------------------------------------------------------
    @property
    def inserted_lengths(self) -> List[float]:
        return self._inserted_lengths

    # ------------------------------------------------------------------
    # 当前旋转角（只读）
    # ------------------------------------------------------------------
    @property
    def rotations(self) -> List[float]:
        return self._rotations

    def close(self):
        self._unload_simulation()

    # ------------------------------------------------------------------
    # 卸载 SOFA 仿真树
    # ------------------------------------------------------------------
    def _unload_simulation(self):
        if self.root is not None:
            self._sofa.Simulation.unload(self.root)

    # ------------------------------------------------------------------
    # 执行动作：action + duration
    # action[:,0] = 插入速度
    # action[:,1] = 旋转速度
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray, duration: float):
        # 将连续时间转换为离散仿真步数
        n_steps = int(duration / self.dt_simulation)
        # === wall-time watchdog ===
        t_start = time.perf_counter()
        max_step_time = 10
        for k in range(n_steps):
            # 当前各器械插入长度
            inserted_lengths = self.inserted_lengths
            if time.perf_counter() - t_start > max_step_time:
                self.logger.warning(
                    f"SOFA step timeout after {k}/{n_steps} substeps "
                    f"(>{max_step_time:.2f}s). Aborting step."
                )
                self.simulation_error = True
                self.reset_devices()  # 或者不 reset，让上层处理
                break
            # 若有多个器械，防止插入交叉（只允许一个领先）
            if len(inserted_lengths) > 1:
                max_id = np.argmax(inserted_lengths)
                new_length = inserted_lengths + action[:, 0] * self.dt_simulation
                new_max_id = np.argmax(new_length)
                if max_id != new_max_id:
                    if abs(action[max_id, 0]) > abs(action[new_max_id, 0]):
                        action[new_max_id, 0] = 0.0
                    else:
                        action[max_id, 0] = 0.0

            # 读取 SOFA 控制器中的插入位置
            x_tip = self._instruments_combined.m_ircontroller.xtip
            # 读取旋转角
            tip_rot = self._instruments_combined.m_ircontroller.rotationInstrument
            # 对每个器械施加动作
            for i in range(action.shape[0]):
                x_tip[i] += float(action[i][0] * self.root.dt.value)
                tip_rot[i] += float(action[i][1] * self.root.dt.value)

            # 写回 SOFA 控制器
            self._instruments_combined.m_ircontroller.xtip = x_tip
            self._instruments_combined.m_ircontroller.rotationInstrument = tip_rot
            # 推进 SOFA 一步仿真，这一步卡死怎么办？
            self._sofa.Simulation.animate(self.root, self.root.dt.value)

            # print(f'simulation_error={self.simulation_error}')
            # 立刻检查是否出现“不收敛” warning
            if self._lcp_no_convergence_seen:
                self.logger.warning("Detected SOFA LCP 'No convergence' warning. Resetting devices.")
                self._lcp_no_convergence_seen = False  # 清一次，避免连环触发
                self.simulation_error = True
                print(f'simulation_error={self.simulation_error}')
                self.reset_devices()
                break

        # 更新 Python 侧缓存状态
        self._update_properties()

    # ------------------------------------------------------------------
    # 重置器械状态（不重建场景）：把插入长度清零、旋转随机化、并 reset SOFA
    # ------------------------------------------------------------------
    def reset_devices(self):
        # 读取当前各器械的插入长度（xtip），这是 SOFA Data 的 value
        x = self._instruments_combined.m_ircontroller.xtip.value
        # 将插入长度全部置 0（x * 0.0 保持 shape 一致）
        self._instruments_combined.m_ircontroller.xtip.value = x * 0.0
        # 读取当前各器械的旋转角数组
        ri = self._instruments_combined.m_ircontroller.rotationInstrument.value
        # 生成与 ri 同 shape 的随机数，并映射到 [0, 2π)
        ri = self._rng.random(ri.shape) * 2 * np.pi
        # 写回 SOFA 控制器：随机初始化每个器械的旋转
        self._instruments_combined.m_ircontroller.rotationInstrument.value = ri
        # 将“第一节点索引”重置为 0（通常表示器械从第 0 个节点开始约束/插入）
        self._instruments_combined.m_ircontroller.indexFirstNode.value = 0
        # 调用 SOFA 的 reset：回到初始状态（包含动力学状态、约束等）
        self._sofa.Simulation.reset(self.root)
        # reset 后同步更新 Python 侧缓存（位置/插入长度/旋转）
        self._update_properties()

    # ------------------------------------------------------------------
    # reset：根据插入点/方向、血管 mesh、器械列表等配置初始化或重建 SOFA 场景
    # 只有当关键配置发生变化时才会“卸载并重建”，否则只更新属性缓存
    # ------------------------------------------------------------------
    def reset(
        self,
        insertion_point,
        insertion_direction,
        mesh_path,
        devices: List[Device],
        coords_high: Optional[Tuple[float, float, float]] = None,
        coords_low: Optional[Tuple[float, float, float]] = None,
        vessel_visual_path: Optional[str] = None,
        seed: int = None,
    ):
        import os
        print("SIM PID =", os.getpid())

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._sofa is None:
            self._sofa = importlib.import_module("Sofa")
        if self._sofa_runtime is None:
            self._sofa_runtime = importlib.import_module("SofaRuntime")
        self._install_sofa_log_watch()
        self.simulation_error = False
        # ------------------------------------------------------------------
        # 判断是否需要重建场景：
        # 只要 root 为空或关键配置变化，就卸载并重新搭建 SOFA 场景树
        # ------------------------------------------------------------------
        if (
            self.root is None
            or np.any(insertion_point != self._insertion_point)
            or np.any(insertion_direction != self._insertion_direction)
            or mesh_path != self._mesh_path
            or vessel_visual_path != self._vessel_visual_path
            or np.any(coords_high != self._coords_high)
            or np.any(coords_low != self._coords_low)
            # or self.init_visual_nodes
        ):
            if self.root is None:
                self.root = self._sofa.Core.Node()
            else:
                self._unload_simulation()

            self.root.gravity = [0.0, 0.0, 0.0]
            self.root.dt = self.dt_simulation
            self._load_plugins()
            self._basic_setup(self.friction)
            self._add_vessel_tree(mesh_path=mesh_path)
            self._add_devices(
                devices=devices,
                insertion_point=insertion_point,
                insertion_direction=insertion_direction,
            )
            if self.init_visual_nodes:
                self._add_visual(
                    self.display_size,
                    coords_low,
                    coords_high,
                    self.target_size,
                    self.interim_target_size,
                    devices=devices,
                    vessel_visual_path=vessel_visual_path,
                )

            self._sofa.Simulation.init(self.root)
            self._insertion_point = insertion_point
            self._insertion_direction = insertion_direction
            self._mesh_path = mesh_path
            self._coords_high = coords_high
            self._coords_low = coords_low
            self._vessel_visual_path = vessel_visual_path
            self.simulation_error = False
            self.logger.debug("Sofa Initialized")
        self._update_properties()

    # ------------------------------------------------------------------
    # 更新缓存属性：从 SOFA 里取出当前 DOF 位置、插入长度、旋转角
    # 并检测 NaN（仿真发散）后自动 reset_devices()
    # ------------------------------------------------------------------
    def _update_properties(self) -> None:
        tracking = self._instruments_combined.DOFs.position.value[:, 0:3][::-1]
        if np.any(np.isnan(tracking[0])):
            self.logger.warning("Tracking is NAN, resetting devices")
            print(f'simulation_error={self.simulation_error}')
            self.simulation_error = True
            self.reset_devices()
            tracking = self._instruments_combined.DOFs.position.value[:, 0:3][::-1]
        self._dof_positions = deepcopy(tracking)
        self._inserted_lengths = deepcopy(
            self._instruments_combined.m_ircontroller.xtip.value
        )
        self._rotations = deepcopy(
            self._instruments_combined.m_ircontroller.rotationInstrument.value
        )

    # ------------------------------------------------------------------
    # 加载 SOFA 所需插件（BeamAdapter + 碰撞 + 求解器 + 拓扑映射等）
    # ------------------------------------------------------------------
    def _load_plugins(self):
        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            BeamAdapter\
            Sofa.Component.AnimationLoop\
            Sofa.Component.Collision.Detection.Algorithm\
            Sofa.Component.Collision.Detection.Intersection\
            Sofa.Component.LinearSolver.Direct\
            Sofa.Component.IO.Mesh\
            Sofa.Component.ODESolver.Backward\
            Sofa.Component.Constraint.Lagrangian.Correction\
            Sofa.Component.Topology.Mapping",
        )

    # ------------------------------------------------------------------
    # 场景基础设置：动画循环 + 碰撞管线 + 接触距离 + 摩擦接触约束求解
    # ------------------------------------------------------------------
    def _basic_setup(self, friction: float):
        self.root.addObject("FreeMotionAnimationLoop")
        self.root.addObject("DefaultPipeline", draw="0", depth="6", verbose="1")
        self.root.addObject("BruteForceBroadPhase")
        self.root.addObject("BVHNarrowPhase")
        self.root.addObject(
            "LocalMinDistance",
            contactDistance=0.1,
            alarmDistance=0.2,
            angleCone=0.02,
            name="localmindistance",
        )
        self.root.addObject(
            "DefaultContactManager", response="FrictionContactConstraint"
        )
        self.root.addObject(
            "LCPConstraintSolver",
            mu=friction,
            tolerance=1e-4,
            maxIt=2000,
            name="LCP",
            build_lcp=False,
        )

    # ------------------------------------------------------------------
    # 添加血管树：把血管 mesh 加载为“静态碰撞体”
    # ------------------------------------------------------------------
    def _add_vessel_tree(self, mesh_path):
        vessel_object = self.root.addChild("vesselTree")
        vessel_object.addObject(
            "MeshObjLoader",
            filename=mesh_path,
            flipNormals=False,
            name="meshLoader",
        )
        vessel_object.addObject(
            "MeshTopology",
            position="@meshLoader.position",
            triangles="@meshLoader.triangles",
        )
        vessel_object.addObject("MechanicalObject", name="dofs", src="@meshLoader")
        vessel_object.addObject("TriangleCollisionModel", moving=False, simulated=False)
        vessel_object.addObject("LineCollisionModel", moving=False, simulated=False)
        self._vessel_object = vessel_object

    # ------------------------------------------------------------------
    # 添加器械（多器械）：为每个器械创建 rest shape + 拓扑容器
    # 然后在 InstrumentCombined 下统一建立 DOFs、插值、力场、控制器、碰撞映射
    # ------------------------------------------------------------------
    def _add_devices(self, devices: List[Device], insertion_point, insertion_direction):
        # ==============================================================
        # A) 每个 device：创建 topolines_* 节点 + WireRestShape + 线拓扑容器
        # ==============================================================
        for device in devices:
            # 取该 device 的 sofa_device（包含几何离散、材料参数等）
            sofa_device = device.sofa_device
            topo_lines = self.root.addChild("topolines_" + device.name)
            # 如果该器械不是“程序化生成”，则需要从 OBJ 折线加载中心线
            if not sofa_device.is_a_procedural_shape:
                topo_lines.addObject(
                    "MeshObjLoader",
                    filename=device.sofa_device.mesh_path,
                    name="loader",
                )
            # WireRestShape：BeamAdapter 的核心组件之一
            # 定义器械的参考形状（中心线）、分段关键点、beam 密度、可视化/碰撞离散等
            topo_lines.addObject(
                "WireRestShape",
                name="rest_shape_" + device.name,
                isAProceduralShape=sofa_device.is_a_procedural_shape,
                straightLength=sofa_device.straight_length,
                length=sofa_device.length,
                spireDiameter=sofa_device.spire_diameter,
                radiusExtremity=sofa_device.radius_extremity,
                youngModulusExtremity=sofa_device.young_modulus_extremity,
                massDensityExtremity=sofa_device.mass_density_extremity,
                radius=sofa_device.radius,
                youngModulus=sofa_device.young_modulus,
                massDensity=sofa_device.mass_density,
                poissonRatio=sofa_device.poisson_ratio,
                keyPoints=sofa_device.key_points,
                densityOfBeams=sofa_device.density_of_beams,
                numEdgesCollis=sofa_device.num_edges_collis,
                numEdges=sofa_device.num_edges,
                spireHeight=sofa_device.spire_height,
                printLog=True,
                template="Rigid3d",
            )
            topo_lines.addObject(
                "EdgeSetTopologyContainer", name="meshLines_" + device.name
            )
            topo_lines.addObject("EdgeSetTopologyModifier", name="Modifier")
            topo_lines.addObject(
                "EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d"
            )
            topo_lines.addObject(
                "MechanicalObject", name="dofTopo_" + device.name, template="Rigid3d"
            )

        # ==============================================================
        # B) 合并器械节点：InstrumentCombined
        # 统一管理所有器械的 DOFs、求解器、控制器、碰撞
        # ==============================================================
        instruments_combined = self.root.addChild("InstrumentCombined")
        # 隐式积分器：EulerImplicit（更稳定，适合刚度较大系统）
        instruments_combined.addObject(
            "EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1
        )
        # 线性求解器：BTDLinearSolver（BeamAdapter 常用）
        instruments_combined.addObject(
            "BTDLinearSolver", verification=False, subpartSolve=False, verbose=False
        )
        # 统计总 beam 数（用于创建统一 DOFs 拓扑长度）
        nx = 0
        for device in devices:
            # device.sofa_device.density_of_beams 可能是分段列表，这里 sum 后累加
            nx = sum([nx, sum(device.sofa_device.density_of_beams)])

        # RegularGridTopology：创建一个规则网格拓扑（这里用作“线性序列点”的容器）
        instruments_combined.addObject(
            "RegularGridTopology",
            name="MeshLines",
            nx=nx + 1,
            ny=1,
            nz=1,
            xmax=1.0,
            xmin=0.0,
            ymin=0,
            ymax=0,
            zmax=1,
            zmin=1,
            p0=[0, 0, 0],
        )
        # DOFs：统一的刚体自由度数组（Rigid3d）
        instruments_combined.addObject(
            "MechanicalObject",
            showIndices=False,
            name="DOFs",
            template="Rigid3d",
        )
        # 初始化每个器械的插入长度列表（xtip）
        x_tip = []
        # 初始化每个器械的旋转角列表（rotationInstrument）
        rotations = []
        # instruments 字符串：InterventionalRadiologyController 需要以空格分隔的插值器名字
        interpolations = ""

        # ==============================================================
        # C) 对每个器械：添加插值器 + 力场，并收集 xtip/rotation/instruments
        # ==============================================================
        for device in devices:
            # WireRestShape 的路径引用（跨节点引用：topolines_* 里 rest_shape_*）
            wire_rest_shape = (
                "@../topolines_" + device.name + "/rest_shape_" + device.name
            )
            # WireBeamInterpolation：把 rest shape + xtip 等映射成 beam 的插值信息
            instruments_combined.addObject(
                "WireBeamInterpolation",
                name="Interpol_" + device.name,
                WireRestShape=wire_rest_shape,
                radius=device.sofa_device.radius,
                printLog=False,
            )
            # AdaptiveBeamForceFieldAndMass：Beam 的力场与质量（弯曲、拉伸等）
            instruments_combined.addObject(
                "AdaptiveBeamForceFieldAndMass",
                name="ForceField_" + device.name,
                massDensity=device.sofa_device.mass_density,
                interpolation="@Interpol_" + device.name,
            )
            # 初始插入长度为 0
            x_tip.append(0.0)
            # 初始旋转角随机 [0,2π)
            rotations.append(self._rng.random() * math.pi * 2)
            # instruments 字符串追加当前插值器名（末尾多一个空格，后面会去掉）
            interpolations += "Interpol_" + device.name + " "
        # 让第 0 个器械初始插入一点点（避免完全重合/接触初始化不稳定）
        x_tip[0] += 0.1
        # 去掉 instruments 字符串末尾多余空格
        interpolations = interpolations[:-1]

        # 计算插入位姿 startingPos（插入点 + 对齐插入方向的四元数）
        insertion_pose = self._calculate_insertion_pose(
            insertion_point, insertion_direction
        )

        # ==============================================================
        # D) 介入控制器：InterventionalRadiologyController
        # 负责把 xtip/rotation 等控制量作用到各个 instruments（插值器）
        # ==============================================================
        instruments_combined.addObject(
            "InterventionalRadiologyController",
            name="m_ircontroller",
            template="Rigid3d",
            instruments=interpolations,
            startingPos=insertion_pose,
            xtip=x_tip,
            printLog=True,
            rotationInstrument=rotations,
            speed=0.0,
            listening=True,
            controlledInstrument=0,
        )

        instruments_combined.addObject(
            "LinearSolverConstraintCorrection", wire_optimization="true", printLog=False
        )
        instruments_combined.addObject(
            "FixedConstraint", indices=0, name="FixedConstraint"
        )
        instruments_combined.addObject(
            "RestShapeSpringsForceField",
            points="@m_ircontroller.indexFirstNode",
            angularStiffness=1e8,
            stiffness=1e8,
            external_points=0,
            external_rest_shape="@DOFs",
        )
        self._instruments_combined = instruments_combined

        beam_collis = instruments_combined.addChild("CollisionModel")
        beam_collis.activated = True
        beam_collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        beam_collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        beam_collis.addObject("MechanicalObject", name="CollisionDOFs")
        beam_collis.addObject(
            "MultiAdaptiveBeamMapping",
            controller="../m_ircontroller",
            useCurvAbs=True,
            printLog=False,
            name="collisMap",
        )
        beam_collis.addObject("LineCollisionModel", proximity=0.0)
        beam_collis.addObject("PointCollisionModel", proximity=0.0)

    # ------------------------------------------------------------------
    # 添加可视化（OpenGL）：血管、器械表面、目标点、相机与光照
    # ------------------------------------------------------------------
    def _add_visual(
        self,
        display_size: Tuple[int, int],
        coords_low: Tuple[float, float, float],
        coords_high: Tuple[float, float, float],
        target_size: float,
        interim_target_size: float,
        devices: List[Device],
        vessel_visual_path: Optional[str] = None,
    ):
        coords_low = np.array(coords_low)
        coords_high = np.array(coords_high)
        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            Sofa.GL.Component.Rendering3D\
            Sofa.GL.Component.Shader",
        )

        # ==============================================================
        # 1) 血管可视化
        # ==============================================================
        # 如果没有提供单独的可视化血管 mesh，就直接用碰撞 mesh 渲染
        # Vessel Tree
        if vessel_visual_path is None:
            self._vessel_object.addObject(
                "OglModel",
                src="@meshLoader",
                color=[1.0, 0.0, 0.0, 0.3],
            )
        # 否则：加载一个独立的血管视觉 mesh，并把它映射到碰撞 dofs 上
        else:
            visu_vessel = self._vessel_object.addChild("Visual Vessel")
            visu_vessel.addObject(
                "MeshObjLoader", name="loader", filename=vessel_visual_path
            )
            visu_vessel.addObject("MechanicalObject", name="visu")
            visu_vessel.addObject(
                "OglModel", name="Visu", src="@loader", color=[1.0, 0.0, 0.0, 0.3]
            )
            visu_vessel.addObject(
                "BarycentricMapping", input="@../dofs", output="@Visu"
            )

        # ==============================================================
        # 2) 器械可视化：把中心线边 → 生成管状 quad 网格 → 显示为 OglModel
        # ==============================================================
        # Devices
        for device in devices:
            # 为每个器械创建可视化节点（挂在 InstrumentCombined 下）
            visu_node = self._instruments_combined.addChild("Visu_" + device.name)
            visu_node.activated = True
            visu_node.addObject("MechanicalObject", name="Quads")
            visu_node.addObject(
                "QuadSetTopologyContainer", name="Container_" + device.name
            )
            visu_node.addObject("QuadSetTopologyModifier", name="Modifier")
            visu_node.addObject(
                "QuadSetGeometryAlgorithms",
                name="GeomAlgo",
                template="Vec3d",
            )
            mesh_lines = "@../../topolines_" + device.name + "/meshLines_" + device.name
            # Edge2QuadTopologicalMapping：把中心线边“挤出”为圆环采样的 quad 管面
            visu_node.addObject(
                "Edge2QuadTopologicalMapping",
                nbPointsOnEachCircle=10,
                radius=device.sofa_device.radius,
                flipNormals="true",
                input=mesh_lines,
                output="@Container_" + device.name,
            )
            visu_node.addObject(
                "AdaptiveBeamMapping",
                interpolation="@../Interpol_" + device.name,
                name="VisuMap_" + device.name,
                output="@Quads",
                isMechanical="false",
                input="@../DOFs",
                useCurvAbs="1",
                printLog="0",
            )
            visu_ogl = visu_node.addChild("VisuOgl")
            visu_ogl.activated = True
            visu_ogl.addObject(
                "OglModel",
                color=device.color,
                quads="@../Container_" + device.name + ".quads",
                material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
                name="Visual",
            )
            visu_ogl.addObject(
                "IdentityMapping",
                input="@../Quads",
                output="@Visual",
            )

        # ==============================================================
        # 3) 主目标点（球体）可视化 + 刚体映射
        # ==============================================================
        # Target
        # TODO: Fix necessary translation of ogl_model. Maybe unite_sphere.obj with center in origin?
        file_dir = os.path.dirname(os.path.realpath(__file__))
        mesh_path = os.path.join(file_dir, "util", "unit_sphere.stl")
        target_node = self.root.addChild("main_target")
        target_node.addObject(
            "MeshSTLLoader",
            name="loader",
            triangulate=True,
            filename=mesh_path,
            scale=target_size,
            translation=[0, 0, 0],
        )
        target_node.addObject(
            "MechanicalObject",
            src="@loader",
            translation=(0, 0, 0),
            template="Rigid3d",
            name="MechanicalObject",
        )
        size_half = target_size / 2
        target_node.addObject(
            "OglModel",
            src="@loader",
            color=[0.0, 0.9, 0.5, 0.8],
            translation=[0, 0, -size_half],
            material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
            name="ogl_model",
        )
        target_node.addObject("RigidMapping", input="@MechanicalObject")
        self.target_node = target_node

        # ==============================================================
        # 4) 中间目标点（最多 100 个）可视化
        # ==============================================================
        self.interim_targets = []
        interim_target_node = self.root.addChild("interim_target")
        interim_target_node.addObject(
            "MeshSTLLoader",
            name="loader",
            triangulate=True,
            filename=mesh_path,
            scale=interim_target_size,
            translation=[0, 0, 0],
        )
        interim_target_node.addObject(
            "MechanicalObject",
            src="@loader",
            translation=(9999, 0, 0),
            template="Rigid3d",
            name="MechanicalObject",
        )
        self.interim_target_node = interim_target_node
        # 预先创建 100 个子节点，每个子节点一个 OglModel（便于后面快速更新位置）
        for i in range(100):
            interim_node = self.interim_target_node.addChild(f"interim_node_{i}")

            interim_node.addObject(
                "OglModel",
                src="@../loader",
                color=[0.0, 0.9, 0.5, 0.2],
                translation=[0.0, 0.0, 0.0],
                material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
                name="ogl_model",
            )
            self.interim_targets.append(interim_node)

        # ==============================================================
        # 5) 相机与光照
        # ==============================================================
        # Camera
        # TODO: Find out how to manipulate background. BackgroundSetting doesn't seem to work
        # self.root.addObject("BackgroundSetting", color=(0.5, 0.5, 0.5, 1.0))
        self.root.addObject("DefaultVisualManagerLoop")
        self.root.addObject(
            "VisualStyle",
            displayFlags="showVisualModels\
                hideBehaviorModels\
                hideCollisionModels\
                hideWireframe\
                hideMappings\
                hideForceFields",
        )

        self.root.addObject("LightManager")
        self.root.addObject("DirectionalLight", direction=[0, -1, 0])
        self.root.addObject("DirectionalLight", direction=[0, 1, 0])

        # ==============================================================
        # 6) 计算相机 lookAt / position / near / far    512*512*512坐标系
        # ==============================================================
        # look_at = (coords_high + coords_low) * 0.5
        look_at = [256, 256, 256]
        # print(f'coords_high={coords_high}, coords_low={coords_low}, look_at={look_at}')
        distance_coefficient = 1.5
        # distance = np.linalg.norm(look_at - coords_low) * distance_coefficient
        distance = 829.4015277/157.7*512
        # print(f'distance={distance}')

        theta = np.deg2rad(float(30) - 90)  # 极角
        phi = np.deg2rad(float(0))  # 极坐标

        x_camera = distance * np.sin(theta) * np.cos(phi)
        y_camera = distance * np.sin(theta) * np.sin(phi)
        z_camera = distance * np.cos(theta)
        print(f'x_camera={x_camera}, y_camera={y_camera}, z_camera={z_camera}')

        position = look_at + np.array([x_camera, y_camera, z_camera])
        print(f'position={position}')
        scene_radius = np.linalg.norm(coords_high - coords_low)
        dist_cam_to_center = np.linalg.norm(position - look_at)
        z_clipping_coeff = 5
        z_near_coeff = 0.01
        z_near = dist_cam_to_center - scene_radius
        z_far = (z_near + 2 * scene_radius) * 2
        z_near = z_near * z_near_coeff
        z_min = z_near_coeff * z_clipping_coeff * scene_radius
        if z_near < z_min:
            z_near = z_min
        field_of_view = 2.0 * np.arctan(0.5 / 7.191) * 180.0 / np.pi
        look_at = np.array(look_at)
        position = np.array(position)
        # print(f'display_size={display_size}')   # 512
        print(f'z_near={z_near}')
        print(f'z_far={z_far}')

        self.camera = self.root.addObject(
            "Camera",
            name="camera",
            lookAt=look_at,
            position=position,
            fieldOfView=field_of_view,
            widthViewport=display_size[0],
            heightViewport=display_size[1],
            zNear=z_near,
            zFar=z_far,
            fixedLookAt=True,
        )

        '''# ==============================================================
        # 5) 相机与光照
        # ==============================================================
        # Camera
        # TODO: Find out how to manipulate background. BackgroundSetting doesn't seem to work
        # self.root.addObject("BackgroundSetting", color=(0.5, 0.5, 0.5, 1.0))
        self.root.addObject("DefaultVisualManagerLoop")
        self.root.addObject(
            "VisualStyle",
            displayFlags="showVisualModels\
                        hideBehaviorModels\
                        hideCollisionModels\
                        hideWireframe\
                        hideMappings\
                        hideForceFields",
        )
        self.root.addObject("LightManager")
        self.root.addObject("DirectionalLight", direction=[0, -1, 0])
        self.root.addObject("DirectionalLight", direction=[0, 1, 0])

        # ==============================================================
        # 6) 计算相机 lookAt / position / near / far
        # ==============================================================
        look_at = (coords_high + coords_low) * 0.5
        distance_coefficient = 1.5
        distance = np.linalg.norm(look_at - coords_low) * distance_coefficient
        position = look_at + np.array([-distance, 0.0, 0.0])
        scene_radius = np.linalg.norm(coords_high - coords_low)
        dist_cam_to_center = np.linalg.norm(position - look_at)
        z_clipping_coeff = 5
        z_near_coeff = 0.01
        z_near = dist_cam_to_center - scene_radius
        z_far = (z_near + 2 * scene_radius) * 2
        z_near = z_near * z_near_coeff
        z_min = z_near_coeff * z_clipping_coeff * scene_radius
        if z_near < z_min:
            z_near = z_min
        field_of_view = 70
        look_at = np.array(look_at)
        position = np.array(position)

        self.camera = self.root.addObject(
            "Camera",
            name="camera",
            lookAt=look_at,
            position=position,
            fieldOfView=field_of_view,
            widthViewport=display_size[0],
            heightViewport=display_size[1],
            zNear=z_near,
            zFar=z_far,
            fixedLookAt=False,
        )'''

    # ------------------------------------------------------------------
    # 更新中间目标点位置：给定 positions 列表，更新前 n_targets 个 OglModel 的平移
    # 多余的 interim_targets 会被删除
    # ------------------------------------------------------------------
    def add_interim_targets(self, positions: List[Tuple[float, float, float]]):
        n_targets = min(len(positions), len(self.interim_targets))
        for i in range(n_targets):
            position = tuple(positions[i])
            self.interim_targets[i].ogl_model.translation = position
        targets_to_remove = self.interim_targets[i + 1 :]
        for target in targets_to_remove:
            self.remove_interim_target(target)

        self._sofa.Simulation.init(self.interim_target_node)
        return self.interim_targets.copy()

    # ------------------------------------------------------------------
    # 删除一个中间目标点节点：同时从 SOFA 场景树移除、并从 Python 列表移除引用
    # ------------------------------------------------------------------
    def remove_interim_target(self, interim_target):
        self.interim_target_node.removeChild(interim_target)
        self.interim_targets.remove(interim_target)

    # ------------------------------------------------------------------
    # 计算器械插入位姿 startingPos
    # 输出格式： [x, y, z, qx, qy, qz, qw]
    # 目标：把“原始方向 original_direction = +X”旋转到 insertion_direction
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_insertion_pose(
        insertion_point: np.ndarray, insertion_direction: np.ndarray
    ):
        insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
        original_direction = np.array([1.0, 0.0, 0.0])
        if np.all(insertion_direction == original_direction):
            w0 = 1.0
            xyz0 = [0.0, 0.0, 0.0]
        elif np.all(np.cross(insertion_direction, original_direction) == 0):
            w0 = 0.0
            xyz0 = [0.0, 1.0, 0.0]
        else:
            half = (original_direction + insertion_direction) / np.linalg.norm(
                original_direction + insertion_direction
            )
            w0 = np.dot(original_direction, half)
            xyz0 = np.cross(original_direction, half)
        xyz0 = list(xyz0)
        pose = list(insertion_point) + list(xyz0) + [w0]
        return pose

    def _install_sofa_log_watch(self) -> None:
        """
        Capture SOFA C++ warnings printed to terminal in real time.
        Sets self._lcp_no_convergence_seen = True when detected.
        """
        if self._log_capture_installed:
            return

        def on_line(s: str):
            if "LCPConstraintSolver" in s and "No convergence" in s:
                self._lcp_no_convergence_seen = True
            if "error =-nan" in s or "error = -nan" in s:
                self._lcp_no_convergence_seen = True

        # 大多数 SOFA warning 走 stderr（fd=2）
        from .util.sofa_log_watch import FdTeeCapture  # 如果你放同文件，就删掉这行 import，直接用类名

        self._stderr_cap = FdTeeCapture(2, on_line).start()
        self._stdout_cap = FdTeeCapture(1, on_line).start()

        # 如果你发现 warning 实际在 stdout（fd=1），把下面这行也打开
        # self._stdout_cap = FdTeeCapture(1, callback=on_line).start()

        self._log_capture_installed = True

    def add_axes_cross(self, center=(256, 256, 256), L=30.0):
        x0, y0, z0 = center
        pts = [
            [x0 - L, y0, z0], [x0 + L, y0, z0],
            [x0, y0 - L, z0], [x0, y0 + L, z0],
            [x0, y0, z0 - L], [x0, y0, z0 + L],
        ]
        n = self.root.addChild("cross_256")
        n.addObject("MechanicalObject", name="dofs", template="Vec3d", position=pts)

        # 直接让 OglModel 画 edges（关键）
        n.addObject("OglModel", name="visu",
                    position="@dofs.position",
                    edges=[[0, 1], [2, 3], [4, 5]],
                    color=[1, 0, 0, 1])
        return n

    def add_marker(self, center=(256, 256, 256), r=5.0):
        n = self.root.addChild("marker_256")
        n.addObject("MechanicalObject", template="Vec3d", position=[list(center)])
        n.addObject("SphereCollisionModel", radius=r, color=[1.0, 1.0, 0.0, 1.0])
        return n





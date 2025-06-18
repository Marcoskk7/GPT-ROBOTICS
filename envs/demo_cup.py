import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import os
import yaml
from pathlib import Path
import sys
import mplib
# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(parent_directory)  # copy_Robotwin目录
sys.path.append(project_root)

# 修改为绝对导入
from envs.utils import *
from envs.camera import Camera
from envs.robot import Robot

class EmptyCupEnvironment:
    def __init__(self, config_path=None):
        """
        初始化空杯子放置环境（单臂Panda机器人）
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.scene = None
        self.robot = None
        self.cameras = None
        self.cup = None
        self.coaster = None
        self.table = None
        self.wall = None
        self.size_dict = []

    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def _default_config(self):
        """默认配置"""
        return {
            'timestep': 1/250,
            'ground_height': 0,
            'static_friction': 0.5,
            'dynamic_friction': 0.5,
            'restitution': 0,
            'ambient_light': [0.5, 0.5, 0.5],
            'shadow': True,
            'direction_lights': [[[0, 0.5, -1], [0.5, 0.5, 0.5]]],
            'point_lights': [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]],
            'camera_xyz': [0.4, 0.22, 1.5],
            'camera_rpy': [0, -0.8, 2.45],
            'table_height': 0.74,
            'table_pose': [0, 0],
            'render_freq': 10
        }

    def setup_scene(self):
        """设置基础场景"""
        # 创建场景（新版本Sapien直接创建场景）
        self.scene = sapien.Scene()
        
        # 设置物理参数
        self.scene.set_timestep(self.config['timestep'])
        self.scene.add_ground(self.config['ground_height'])
        
        # 设置物理材质
        self.scene.default_physical_material = self.scene.create_physical_material(
            self.config['static_friction'],
            self.config['dynamic_friction'],
            self.config['restitution']
        )
        
        # 设置光照
        self.scene.set_ambient_light(self.config['ambient_light'])
        
        # 添加方向光
        for direction_light in self.config['direction_lights']:
            self.scene.add_directional_light(
                direction_light[0], 
                direction_light[1], 
                shadow=self.config['shadow']
            )
        
        # 添加点光源
        for point_light in self.config['point_lights']:
            self.scene.add_point_light(
                point_light[0], 
                point_light[1], 
                shadow=self.config['shadow']
            )

        # 初始化观察器
        if self.config['render_freq']:
            self.viewer = Viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(*self.config['camera_xyz'])
            self.viewer.set_camera_rpy(*self.config['camera_rpy'])

    def create_table_and_wall(self):
        """创建桌子和墙壁"""
        table_pose = self.config['table_pose']
        table_height = self.config['table_height']
        
        # 创建墙壁
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name='wall',
            is_static=True
        )
        
        # 创建桌子
        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_pose[0], table_pose[1], table_height]),
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            is_static=True
        )
    # def load_robot(self, **kwargs):
    #     """加载Panda单臂机器人"""
    #     self.robot = Robot(self.scene, **kwargs)
    #     self.robot.set_planner()
    #     self.robot.init_joints()    
    def load_robot(self, **kwargs):
        """加载Panda单臂机器人"""
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        
        # 默认使用Panda机器人URDF
        urdf_path = kwargs.get("urdf_path", "./assets/embodiments/panda/panda.urdf")
        self.robot = loader.load(urdf_path)
        
        # 设置机器人位置 - 放在桌子上
        table_height = self.config['table_height']  # 0.74
        robot_x = kwargs.get("robot_origin_xyz", [0, -0.3, table_height])[0]  # 桌子中央
        robot_y = -0.3  # 贴近墙壁（墙壁前表面在y=0.4，留一点空间）
        robot_z = table_height  # 桌面高度
    
        self.robot.set_root_pose(
            sapien.Pose(
                [robot_x, robot_y, robot_z],  # 放在桌面上
                kwargs.get("robot_origin_quat", [0.707, 0, 0, 0.707]),
            )
        )
    
        
        # 设置关节驱动属性
        self.active_joints = self.robot.get_active_joints()
        for i, joint in enumerate(self.active_joints):
            if i == 6:  # 第7个关节（索引6）是手腕旋转关节
                # 对手腕旋转关节使用更高的阻尼和更低的刚度，让它更稳定
                joint.set_drive_property(
                    stiffness=kwargs.get("wrist_joint_stiffness", 500),  # 降低刚度
                    damping=kwargs.get("wrist_joint_damping", 500),      # 增加阻尼
                )
                print(f"设置手腕关节 {joint.get_name()} 特殊驱动属性")
            else:
                joint.set_drive_property(
                    stiffness=kwargs.get("joint_stiffness", 1000),
                    damping=kwargs.get("joint_damping", 200),
                )

    def load_cameras(self, **kwargs):
        """加载相机（单臂配置）"""
        # 为单臂机器人准备相机配置
        camera_config = {
            'embodiment_config': {
                'static_camera_list': [
                    {
                        'name': 'head_camera',
                        'position': [0, -0.5, 1.5],
                        'forward': [0, 1, -0.5],
                        'left': [-1, 0, 0]
                    },
                    {
                        'name': 'front_camera',
                        'type': 'D435',
                        'position': [0, -0.45, 0.85],
                        'forward': [0, 1, -0.1],
                        'left': [-1, 0, 0]
                    }
                ]
            },
            'dual_arm': False
        }
        
        # 合并用户提供的配置
        camera_config.update(kwargs)
        
        #self.cameras = Camera(**camera_config)
        self.cameras.load_camera(self.scene)
        
        # 运行一个物理步骤并更新渲染
        self.scene.step()
        self.scene.update_render()

    def load_objects(self):
        """加载真实的杯子和杯垫模型（参考empty_cup_place.py）"""
        tag = np.random.randint(0,2)
        if tag==0:
            self.cup,self.cup_data = rand_create_glb(
                self.scene,
                xlim=[0.15,0.3],
                ylim=[0.0,0.25],  # 原来是[0.3,0.55]，降低0.3后变成[0.0,0.25]
                zlim=[0.8],
                modelname="022_cup",
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )
            cup_pose = self.cup.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.05,0.1],
                ylim=[0.0,0.25],  # 原来是[0.3,0.55]，降低0.3后变成[0.0,0.25]
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )

            while np.sum(pow(cup_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.05,0.1],
                    ylim=[0.0,0.25],  # 原来是[0.3,0.55]，降低0.3后变成[0.0,0.25]
                    zlim=[0.76],
                    rotate_rand=False,
                    qpos=[0.5,0.5,0.5,0.5],
                )
            self.coaster,self.coaster_data = create_obj(
                self.scene,
                pose = coaster_pose,
                modelname="019_coaster",
                convex=True
            )
        else:
            self.cup,self.cup_data = rand_create_glb(
                self.scene,
                xlim=[-0.3,-0.15],
                ylim=[0.0,0.25],  # 原来是[0.3,0.55]，降低0.3后变成[0.0,0.25]
                zlim=[0.8],
                modelname="022_cup",
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )
            cup_pose = self.cup.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.1, 0.05],
                ylim=[0.0,0.25],  # 原来是[0.3,0.55]，降低0.3后变成[0.0,0.25]
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )

            while np.sum(pow(cup_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.1, 0.05],
                    ylim=[0.0,0.25],  # 原来是[0.3,0.55]，降低0.3后变成[0.0,0.25]
                    zlim=[0.76],
                    rotate_rand=False,
                    qpos=[0.5,0.5,0.5,0.5],
                )
            self.coaster, self.coaster_data = create_obj(
                self.scene,
                pose = coaster_pose,
                modelname="019_coaster",
                convex=True
            )
        
        self.cup.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.coaster.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        pose = sapien.core.pysapien.Entity.get_pose(self.cup).p.tolist()
        pose.append(0.08)
        self.size_dict.append(pose)
        pose = sapien.core.pysapien.Entity.get_pose(self.coaster).p.tolist()
        pose.append(0.1)
        self.size_dict.append(pose)
        
    def move_to_pose_fixed_wrist(self, target_pose, fixed_wrist_angle):
        """
        移动到目标位置，但保持手腕旋转角度固定
        
        Args:
            target_pose: 目标位置
            fixed_wrist_angle: 固定的手腕旋转角度
        """
        try:
            # 根据目标位置智能选择关节角度
            target_pos = target_pose.p
            
            print(f"移动到目标位置: {target_pos}，手腕角度固定为: {fixed_wrist_angle:.3f}")
            
            # 基础关节角度
            base_qpos = [0, 0.19, 0.0, -2.62, 0.0, 2.94, fixed_wrist_angle, 0, 0]  # 使用固定的手腕角度
            
            # 根据X位置调整第1关节（基座旋转）
            if target_pos[0] > 0.15:  # 右侧
                base_qpos[0] = -0.8
            elif target_pos[0] < -0.15:  # 左侧
                base_qpos[0] = 0.8
            else:  # 中间
                base_qpos[0] = 0.0
            
            # 根据Y位置调整第2关节（肩部）
            if target_pos[1] > -0.05:  # 前方
                base_qpos[1] = 0.5
            else:  # 后方
                base_qpos[1] = 0.19
            
            # 根据Z高度调整第2和第4关节
            if target_pos[2] > 1.1:  # 很高的位置
                base_qpos[1] -= 0.5  # 肩部向下
                base_qpos[3] += 0.5  # 肘部弯曲更多
            elif target_pos[2] > 0.95:  # 高位置
                base_qpos[1] -= 0.3  # 肩部稍微向下
                base_qpos[3] += 0.2  # 肘部稍微弯曲
            elif target_pos[2] < 0.85:  # 低位置（桌面附近）
                base_qpos[1] += 0.2  # 肩部向上
                base_qpos[3] -= 0.3  # 肘部伸直一些
            
            # 获取当前关节角度
            current_qpos = [joint.get_drive_target() for joint in self.active_joints[:7]]
            
            # 确保手腕角度不变
            base_qpos[6] = fixed_wrist_angle
            
            # 增加插值步数使移动更平滑
            steps = 80
            for step in range(steps):
                alpha = step / (steps - 1)
                interpolated_qpos = []
                for i in range(7):
                    if i == 6:  # 手腕旋转关节
                        interpolated_qpos.append(fixed_wrist_angle)  # 始终保持固定角度
                    else:
                        interpolated_qpos.append(
                            current_qpos[i] * (1 - alpha) + base_qpos[i] * alpha
                        )
                
                for i, joint in enumerate(self.active_joints[:7]):
                    joint.set_drive_target(interpolated_qpos[i])
                
                # 运行物理步骤
                for _ in range(2):
                    self.step()
            
            return True
        except Exception as e:
            print(f"固定手腕移动失败: {e}")
            return False

            
    def move_robot_to_home(self):
        """移动机器人到初始位置"""
        # Panda机器人的home位置（7个关节）
        home_qpos = [0, 0.19, 0.0, -2.62, 0.0, 2.94, 0.79, 0,0]  # 典型的Panda home位置
        
        if len(self.active_joints) >= 7:
            for i, joint in enumerate(self.active_joints[:7]):
                joint.set_drive_target(home_qpos[i])
        
        # 运行几步让机器人到达目标位置
        for _ in range(100):
            self.scene.step()

    def create_environment(self, robot_config=None, camera_config=None):
        """
        创建完整的空杯子放置环境（单臂Panda）
        
        Args:
            robot_config: 机器人配置参数
            camera_config: 相机配置参数
        """
        print("设置基础场景...")
        self.setup_scene()
        
        print("创建桌子和墙壁...")
        self.create_table_and_wall()
        
        print("加载机器人...")
        robot_config = robot_config or {}
        self.load_robot(**robot_config)
        
        print("加载相机...")
        camera_config = camera_config or {}
        #self.load_cameras(**camera_config)
        
        print("移动机器人到初始位置...")
        self.move_robot_to_home()
        
        print("加载杯子和杯垫...")
        self.load_objects()
        
        print("环境创建完成!")
        return self

    def step(self):
        """运行一个仿真步骤"""
        self.scene.step()
        self.scene.update_render()
        
        if hasattr(self, 'viewer') and self.config['render_freq']:
            self.viewer.render()

    def get_observation(self):
        """获取观察数据"""
        if self.cameras:
            self.cameras.update_picture()
            return self.cameras.get_config()
        return None

    def reset(self):
        """重置环境"""
        # 移除现有物体
        if self.cup:
            self.scene.remove_actor(self.cup)
        if self.coaster:
            self.scene.remove_actor(self.coaster)
        
        # 重新加载物体
        self.load_objects()
        
        # 重置机器人
        self.move_robot_to_home()

    def close(self):
        """关闭环境"""
        if hasattr(self, 'viewer'):
            self.viewer.close()

    def reset_scene(self):
        """重置整个场景到初始状态"""
        print("🔄 开始重置场景...")
        
        try:
            # 1. 清理现有对象
            if hasattr(self, 'cup') and self.cup:
                self.scene.remove_actor(self.cup)
                self.cup = None
                print("  - 移除旧杯子")
            
            if hasattr(self, 'coaster') and self.coaster:
                self.scene.remove_actor(self.coaster)
                self.coaster = None
                print("  - 移除旧杯垫")
            
            # 2. 机器人回到初始位置
            print("  - 机器人回到初始位置")
            self.move_robot_to_home()
            
            # 3. 重新加载物体
            print("  - 重新加载物体")
            self.load_objects()
            
            # 4. 等待物理仿真稳定
            print("  - 等待场景稳定...")
            for _ in range(100):  # 仿真100步让物体稳定
                self.scene.step()
            
            print("✅ 场景重置完成!")
            return True
            
        except Exception as e:
            print(f"❌ 场景重置失败: {e}")
            return False
    def get_scene_info(self):
        """获取场景信息"""
        info = {
            "cup_exists": self.cup is not None,
            "coaster_exists": self.coaster is not None,
            "robot_initialized": self.robot is not None
        }
        
        try:
            if self.cup:
                cup_pose = self.get_cup_pose()
                if cup_pose:
                    info["cup_position"] = [cup_pose.p[0], cup_pose.p[1], cup_pose.p[2]]
                else:
                    info["cup_position"] = None
        except:
            info["cup_position"] = None
        
        try:
            if self.coaster:
                coaster_pose = self.get_coaster_pose()
                if coaster_pose:
                    info["coaster_position"] = [coaster_pose.p[0], coaster_pose.p[1], coaster_pose.p[2]]
                else:
                    info["coaster_position"] = None
        except:
            info["coaster_position"] = None
        
        return info

class CupPlacementPlanning(EmptyCupEnvironment):
    """
    基于EmptyCupEnvironment的杯子放置路径规划类
    实现Panda机器人将水杯移到杯垫上的功能
    """
    
    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.planner = None
        
    def setup_planner(self):
        """设置运动规划器"""
        try:
            import mplib
            from mplib import Pose
            
            # 获取机器人的关节信息
            active_joints = self.robot.get_active_joints()
            joint_names = [joint.get_name() for joint in active_joints]
            
            print(f"机器人关节名称: {joint_names}")
            
            # 尝试不同的配置方式
            urdf_path = "./assets/embodiments/panda/panda.urdf"
            srdf_path = "./assets/embodiments/panda/panda.srdf"
            
            # 检查文件是否存在
            import os
            if not os.path.exists(urdf_path):
                print(f"URDF文件不存在: {urdf_path}")
                self.planner = None
                return
            
            # 尝试最简配置
            try:
                print("尝试设置mplib规划器...")
                self.planner = mplib.Planner(
                    urdf=urdf_path,
                    srdf=srdf_path if os.path.exists(srdf_path) else None,
                    move_group="panda_hand",
                )
                print("路径规划器创建成功!")
                
                # 设置基座位置 - 使用正确的Pose对象
                try:
                    robot_pose = self.robot.get_pose()
                    base_pose = Pose(
                        [robot_pose.p[0], robot_pose.p[1], robot_pose.p[2]],
                        [robot_pose.q[0], robot_pose.q[1], robot_pose.q[2], robot_pose.q[3]]
                    )
                    self.planner.set_base_pose(base_pose)
                    print("基座位置设置成功")
                except Exception as e:
                    print(f"设置基座位置失败: {e}")
                    print("继续使用规划器，但基座位置可能不准确")
                    
            except Exception as e:
                print(f"设置路径规划器失败: {e}")
                self.planner = None
                
        except ImportError:
            print("警告: mplib未安装")
            self.planner = None
        except Exception as e:
            print(f"设置路径规划器时出错: {e}")
            self.planner = None

    def move_to_pose(self, target_pose, use_point_cloud=False):
        """
        移动到目标位置
        
        Args:
            target_pose: mplib.Pose 目标姿态
            use_point_cloud: 是否使用点云进行碰撞检测
        """
        if self.planner is None:
            print("规划器未可用，使用简单插值移动")
            return self._simple_move_to_pose(target_pose)
            
        try:
            from mplib import Pose
            
            # 将SimplePose转换为mplib.Pose
            if hasattr(target_pose, 'p') and hasattr(target_pose, 'q'):
                mplib_pose = Pose(target_pose.p, target_pose.q)
            else:
                mplib_pose = target_pose
            
            # 获取当前关节角度
            current_qpos = self.robot.get_qpos()
            
            # 使用正确的API进行路径规划
            result = self.planner.plan_pose(
                mplib_pose, 
                current_qpos, 
                time_step=1.0/250.0
            )
            
            if result['status'] != "Success":
                print(f"路径规划失败: {result['status']}")
                return self._simple_move_to_pose(target_pose)
                
            # 执行路径 - 使用示例代码中的follow_path方法
            self.follow_path(result)
            return True
            
        except Exception as e:
            print(f"路径规划执行失败: {e}")
            return self._simple_move_to_pose(target_pose)
        
    def follow_path(self, result):
        """Helper function to follow a path generated by the planner"""
        n_step = result["position"].shape[0]
        
        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            
            # 设置关节位置和速度
            for j in range(len(self.planner.move_group_joint_indices)):
                self.active_joints[j].set_drive_target(result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(result["velocity"][i][j])
            
            # 仿真步骤
            self.scene.step()
            
            # 每4个仿真步骤渲染一次
            if i % 4 == 0:
                self.scene.update_render()
                if hasattr(self, 'viewer') and self.viewer is not None:
                    self.viewer.render()

    def reset_and_reinitialize(self):
        """重置并重新初始化整个环境"""
        print("🔄 完全重置环境...")
        
        try:
            # 重置场景
            success = self.reset_scene()
            if not success:
                return False
            
            # 重新设置规划器（如果需要的话）
            if hasattr(self, 'planner') and self.planner:
                print("  - 重新初始化路径规划器")
                self.setup_planner()
            
            # 显示重置后的状态
            print("  - 重置后场景信息:")
            scene_info = self.get_scene_info()
            for key, value in scene_info.items():
                print(f"    {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ 环境重置失败: {e}")
            return False

    def get_cup_pose(self):
        """获取杯子的当前位置"""
        if self.cup is None:
            return None
        cup_pose = self.cup.get_pose()
        
        # 创建简单的位置对象而不依赖mplib.Pose
        class SimplePose:
            def __init__(self, position, quaternion):
                self.p = position
                self.q = quaternion
        
        return SimplePose(cup_pose.p.tolist(), cup_pose.q.tolist())

    def get_coaster_pose(self):
        """获取杯垫的当前位置"""
        if self.coaster is None:
            return None
        coaster_pose = self.coaster.get_pose()
        
        # 创建简单的位置对象而不依赖mplib.Pose
        class SimplePose:
            def __init__(self, position, quaternion):
                self.p = position
                self.q = quaternion
        
        return SimplePose(coaster_pose.p.tolist(), coaster_pose.q.tolist())

    def plan_cup_to_coaster(self):
        """
        规划将杯子移动到杯垫上的完整动作序列
        """
        print("开始规划杯子到杯垫的路径...")
        
        # 获取杯子和杯垫的位置
        cup_pose = self.get_cup_pose()
        coaster_pose = self.get_coaster_pose()
        
        if cup_pose is None or coaster_pose is None:
            print("无法获取杯子或杯垫的位置")
            return False
        
        print(f"杯子位置: {cup_pose.p}")
        print(f"杯垫位置: {coaster_pose.p}")
        
        # 创建简单的目标位置对象
        class SimplePose:
            def __init__(self, position, quaternion=[0, 1, 0, 0]):
                self.p = position
                self.q = quaternion
        ####
        # 应改为先打开爪子再定位
        ####
        try:
            # 确定抓取方向和偏移
            gripper_offset = 0.02  # 夹爪偏移距离
            x_offset = gripper_offset
            print("使用左侧夹爪对准杯子圆心")
            # 1. 移动到杯子上方（预抓取位置）- 左侧夹爪对齐杯子圆心
            print("步骤1: 移动到杯子上方...")
            pre_grasp_pose = SimplePose([
                cup_pose.p[0] ,  
                cup_pose.p[1] +0.05, 
                cup_pose.p[2] + 0.18
            ])
            if not self.move_to_pose(pre_grasp_pose):
                print("移动到预抓取位置失败")
                return False
            
            # 1. 打开夹爪
            print("步骤2: 打开夹爪...")
            self.open_gripper(0.4)           

            
            # 3. 下降到杯子位置 - 保持左侧夹爪对齐
            print("步骤3: 下降到杯子...")
            grasp_pose = SimplePose([
                cup_pose.p[0],  # 保持相同的偏移
                cup_pose.p[1]+0.05, 
                cup_pose.p[2] + 0.12
            ])
            if not self.move_to_pose(grasp_pose):
                print("下降到抓取位置失败")
                return False
            
            # 4. 关闭夹爪抓取杯子
            print("步骤4: 抓取杯子...")
            self.close_gripper()
            
            # 5. 提升杯子
            print("步骤5: 提升杯子...")
            lift_pose = SimplePose([cup_pose.p[0], cup_pose.p[1], cup_pose.p[2] + 0.18])
            if not self.move_to_pose(lift_pose):
                print("提升杯子失败")
                return False
            
            # 6. 移动到杯垫上方
            print("步骤6: 移动到杯垫上方...")
            pre_place_pose = SimplePose([coaster_pose.p[0], 
                                         coaster_pose.p[1]+0.05, 
                                         coaster_pose.p[2] + 0.18])
            if not self.move_to_pose(pre_place_pose):
                print("移动到杯垫上方失败")
                return False
            
            # 7. 下降到杯垫
            print("步骤7: 将杯子放到杯垫上...")
            place_pose = SimplePose([coaster_pose.p[0],
                                      coaster_pose.p[1]+0.05,
                                      coaster_pose.p[2] + 0.15])
            if not self.move_to_pose(place_pose):
                print("放置杯子失败")
                return False
            
            # 8. 释放杯子
            print("步骤8: 释放杯子...")
            self.open_gripper(0.2)
            
            # 9. 移动到杯垫上方
            print("步骤9: 移动到杯垫上方...")
            pre_place_pose = SimplePose([coaster_pose.p[0], 
                                         coaster_pose.p[1]+0.05, 
                                         coaster_pose.p[2] + 0.28])
            if not self.move_to_pose(pre_place_pose):
                print("移动到杯垫上方失败")
                return False

            # # 9. 撤回到安全位置
            # print("步骤9: 撤回到安全位置...")
            # retreat_pose = SimplePose([coaster_pose.p[0], coaster_pose.p[1], coaster_pose.p[2] + 0.18])
            # if not self.move_to_pose(retreat_pose):
            #     print("撤回失败")
            #     return False
            
            return True
            
        except Exception as e:
            print(f"执行放置序列时出错: {e}")
            return False
        
    def set_gripper(self, pos):
        """
        Helper function to activate gripper joints
        Args:
            pos: position of the gripper joint in real number
        """
        # The following two lines are particular to the panda robot
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(pos)
        # 100 steps is plenty to reach the target position
        for i in range(100):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                if hasattr(self, 'viewer') and self.viewer is not None:
                    self.viewer.render()

    def open_gripper(self,x:int):
        """打开夹爪"""
        self.set_gripper(x)

    def close_gripper(self):
        """关闭夹爪"""
        self.set_gripper(0)   
        
    def create_environment_with_planning(self, robot_config=None, camera_config=None):
        """
        创建包含路径规划功能的完整环境
        """
        # 创建基础环境
        self.create_environment(robot_config, camera_config)
        
        # 设置路径规划器
        print("设置路径规划器...")
        self.setup_planner()
        
        return self

    def debug_positions(self):
        """打印所有重要位置信息"""
        print("=== 调试位置信息 ===")
        
        # 机器人基座位置
        robot_pose = self.robot.get_pose()
        print(f"机器人基座位置: {robot_pose.p}")
        print(f"机器人基座旋转: {robot_pose.q}")
        
        # 夹爪末端位置（需要计算正向运动学）
        if len(self.active_joints) >= 7:
            current_qpos = [joint.get_drive_target() for joint in self.active_joints[:7]]
            print(f"当前关节角度: {current_qpos}")
        
        # 夹爪状态
        if len(self.active_joints) >= 9:
            gripper_pos = [self.active_joints[-2].get_drive_target(), 
                        self.active_joints[-1].get_drive_target()]
            print(f"夹爪位置: {gripper_pos}")
        
        # 杯子位置
        if self.cup:
            cup_pose = self.cup.get_pose()
            print(f"杯子位置: {cup_pose.p}")
        
        # 杯垫位置
        if self.coaster:
            coaster_pose = self.coaster.get_pose()
            print(f"杯垫位置: {coaster_pose.p}")
        
        # 桌子信息
        print(f"桌子高度: {self.config['table_height']}")
        
        print("==================")

def main():#
    """示例使用方法"""
    # 创建环境
    env = CupPlacementPlanning()
    
    # 配置机器人参数（单臂Panda）
    robot_config = {
        'urdf_path': './assets/embodiments/panda/panda.urdf',
        'joint_stiffness': 1000,
        'joint_damping': 200,
        'wrist_joint_stiffness': 1500,   # 手腕关节较低刚度
        'wrist_joint_damping': 500,     # 手腕关节较高阻尼
    }
    
    # 配置相机参数
    camera_config = {}
    
    epochs = 5
    for i in range(epochs):
        # 创建完整环境
        success_epochs = 0
        env.create_environment_with_planning(robot_config)
        env.debug_positions()
        try:
            success = env.plan_cup_to_coaster()
            if success:
                print("任务执行成功!")
                success_epochs+=1
            else:
                print("任务执行失败!")
                
            # 继续运行仿真以观察结果
            print("继续运行仿真...")
            for i in range(500):
                env.step()
                if i % 100 == 0:
                    print(f"仿真步骤: {i}")
                    
        except KeyboardInterrupt:
            print("仿真被用户中断")
        except Exception as e:
            print(f"执行过程中出现错误: {e}")
        finally:
            env.close()
    print(f'成功率有{success_epochs/epochs:.2f}')

if __name__ == "__main__":
    main()
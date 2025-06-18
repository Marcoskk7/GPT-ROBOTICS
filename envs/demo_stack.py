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
        # ...existing code...
        
        # 初始化禁止区域列表
        self.prohibited_area = []
        
        def create_block_data(half_size):
            contact_discription_list = []
            contact_points_list = [
                    [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(front)
                    # [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(right)
                    # [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(left)
                    # [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # top_down(back)
                ]
            functional_matrix = np.eye(4)
            functional_matrix[:3,:3] = t3d.euler.euler2mat(np.pi,0,0)
            functional_matrix[:3,3] = np.array([0,0,-half_size[2]])

            data = {
                'center': [0,0,0],
                'extents': half_size,
                'scale': [1,1,1],                                     # scale
                'target_pose': [[[1,0,0,0],[0,1,0,0],[0,0,1,half_size[2]],[0,0,0,1]]],              # target points matrix
                'contact_points_pose' : contact_points_list,    # contact points matrix list
                'transform_matrix': np.eye(4).tolist(),           # transform matrix
                "functional_matrix": [functional_matrix.tolist()],         # functional points matrix
                'contact_points_discription': contact_discription_list,    # contact points discription
                'contact_points_group': [[0, 1, 2, 3]],
                'contact_points_mask': [True],
                'target_point_discription': ["The top surface center of the block." ],
                'functional_point_discription': ["Point0: The center point on the bottom of the block, and functional axis is vertical bottom side down"]
            }

            return data
        
        block_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.76],
            qpos=[1,0,0,0],  # 固定四元数，保持水平
            ylim_prop=True,
            rotate_rand=False,  # 禁用随机旋转
            rotate_lim=[0,0,0.],  # 旋转限制设为0
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[1,0,0,0],  # 固定四元数
                ylim_prop=True,
                rotate_rand=False,  # 禁用随机旋转
                rotate_lim=[0,0,0.],
            )

        self.block1 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(1,0,0),
            name="block1"
        )

        # 创建第二个方块 - 同样禁用随机旋转
        block_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.76],
            qpos=[1,0,0,0],  # 固定四元数
            ylim_prop=True,
            rotate_rand=False,  # 禁用随机旋转
            rotate_lim=[0,0,0.],
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - self.block1.get_pose().p[:2],2)) < 0.01 \
            or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[1,0,0,0],  # 固定四元数
                ylim_prop=True,
                rotate_rand=False,  # 禁用随机旋转
                rotate_lim=[0,0,0.],
            )

        self.block2 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(0,1,0),
            name="block2"
        )

        # 创建第三个方块 - 同样禁用随机旋转
        block_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.76],
            qpos=[1,0,0,0],  # 固定四元数
            ylim_prop=True,
            rotate_rand=False,  # 禁用随机旋转
            rotate_lim=[0,0,0.],
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - self.block1.get_pose().p[:2],2)) < 0.01 or \
            np.sum(pow(block_pose.p[:2] - self.block2.get_pose().p[:2],2)) < 0.01 or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[1,0,0,0],  # 固定四元数
                ylim_prop=True,
                rotate_rand=False,  # 禁用随机旋转
                rotate_lim=[0,0,0.],
            )

        self.block3 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(0,0,1),
            name="block3"
        )

        # 设置方块数据和物理属性
        self.block1_data = self.block2_data = self.block3_data = create_block_data([0.025,0.025,0.025])

        self.block1.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.block2.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.block3.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        # 记录禁止区域
        pose = self.block1.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04])
        pose = self.block2.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04]) 
        pose = self.block3.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04])
        
    # def _simple_move_to_pose(self, target_pose, fixed_wrist_angle=None):
    #     """简单插值移动，支持固定手腕角度"""
    #     try:
    #         target_pos = target_pose.p
            
    #         # 获取当前关节角度
    #         current_qpos = [float(joint.get_drive_target()) for joint in self.active_joints[:7]]
            
    #         # 设置目标关节角度（简化版本）
    #         target_qpos = current_qpos.copy()
            
    #         # 根据目标位置调整关节
    #         target_qpos[0] = -target_pos[0] * 2.0  # 基座旋转
    #         target_qpos[1] = 0.5 + (target_pos[2] - 0.8) * 1.5  # 肩部
    #         target_qpos[3] = -2.0 - (target_pos[2] - 0.8) * 0.8  # 肘部
            
    #         # 固定手腕角度
    #         if fixed_wrist_angle is not None:
    #             target_qpos[6] = float(fixed_wrist_angle)
            
    #         # 平滑移动
    #         for step in range(100):
    #             alpha = step / 99.0
    #             for i in range(7):
    #                 if i == 6 and fixed_wrist_angle is not None:
    #                     # 手腕角度保持固定
    #                     interpolated = fixed_wrist_angle
    #                 else:
    #                     interpolated = current_qpos[i] * (1 - alpha) + target_qpos[i] * alpha
    #                 self.active_joints[i].set_drive_target(interpolated)
                
    #             self.step()
            
    #         return True
            
    #     except Exception as e:
    #         print(f"简单移动失败: {e}")
    #         return False
            
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
        
        print("加载方块...")
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


    def get_block_pose(self, block_name):
        """获取指定方块的当前位置"""
        block = getattr(self, block_name, None)
        if block is None:
            return None
        
        block_pose = block.get_pose()
        
        class SimplePose:
            def __init__(self, position, quaternion):
                self.p = position
                self.q = quaternion
        
        return SimplePose(block_pose.p.tolist(), block_pose.q.tolist())

    def get_block1_pose(self):
        """获取方块1的位置"""
        return self.get_block_pose('block1')

    def get_block2_pose(self):
        """获取方块2的位置"""
        return self.get_block_pose('block2')

    def get_block3_pose(self):
        """获取方块3的位置"""
        return self.get_block_pose('block3')

    def plan_block_stacking(self):
        """
        规划将三个方块进行叠加的完整动作序列
        策略：将block2和block3依次叠加到block1上
        """
        print("开始规划方块叠加任务...")
        
        # 获取所有方块的位置
        block1_pose = self.get_block1_pose()
        block2_pose = self.get_block2_pose()
        block3_pose = self.get_block3_pose()
        
        if not all([block1_pose, block2_pose, block3_pose]):
            print("无法获取所有方块的位置")
            return False
        
        print(f"方块1位置: {block1_pose.p}")
        print(f"方块2位置: {block2_pose.p}")
        print(f"方块3位置: {block3_pose.p}")
        
        # 创建简单的目标位置对象
        class SimplePose:
            def __init__(self, position, quaternion=[0, 1, 0, 0]):
                self.p = position
                self.q = quaternion
        
        try:
            # 方块尺寸信息
            block_half_size = 0.025  # 方块半尺寸
            block_height = block_half_size * 2  # 方块完整高度
            
            # 记录初始手腕角度 - 修复
            initial_qpos = []
            for joint in self.active_joints[:7]:
                drive_target = joint.get_drive_target()
                if hasattr(drive_target, 'item'):
                    initial_qpos.append(drive_target.item())
                else:
                    initial_qpos.append(float(drive_target))
            
            wrist_rotation_angle = initial_qpos[6] if len(initial_qpos) > 6 else 0.79
            
            print(f"锁定手腕旋转角度为: {wrist_rotation_angle:.3f}")
            # === 第一阶段：将block2叠加到block1上 ===
            print("\n=== 阶段1：将方块2叠加到方块1上 ===")

            # 1. 移动到block2上方
            print("步骤1: 移动到方块2上方...")
            above_block2_pose = SimplePose([
                block2_pose.p[0],
                block2_pose.p[1], 
                block2_pose.p[2] + 0.20
            ])
            if not self.move_to_pose(above_block2_pose, wrist_rotation_angle):
                print("移动到方块2上方失败")
                return False        
                
            # 2. 打开夹爪
            print("步骤2: 打开夹爪...")
            self.open_gripper(0.06)
            
            # 3. 下降抓取block2
            print("步骤3: 下降抓取方块2...")
            grasp_block2_pose = SimplePose([
                block2_pose.p[0],
                block2_pose.p[1], 
                block2_pose.p[2] + 0.10  # 在方块中心稍上方
            ])
            if not self.move_to_pose(grasp_block2_pose, wrist_rotation_angle):
                print("下降到方块2失败")
                return False
            
            # 4. 夹取block2
            print("步骤4: 夹取方块2...")
            self.close_gripper()
            
            # 5. 提升block2
            print("步骤5: 提升方块2...")
            lift_block2_pose = SimplePose([
                block2_pose.p[0],
                block2_pose.p[1], 
                block2_pose.p[2] + 0.25
            ])
            if not self.move_to_pose(lift_block2_pose, wrist_rotation_angle):
                print("提升方块2失败")
                return False
            
            # 6. 移动到block1上方
            print("步骤6: 移动到方块1上方...")
            above_block1_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.2
            ])
            if not self.move_to_pose(above_block1_pose, wrist_rotation_angle):
                print("移动到方块1上方失败")
                return False
            
            # 7. 将block2放置到block1上
            print("步骤7: 将方块2放置到方块1上...")
            place_block2_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.15 # block1顶部 + block2半高度
            ])
            if not self.move_to_pose(place_block2_pose, wrist_rotation_angle):
                print("放置方块2失败")
                return False
            
            # 8. 释放block2
            print("步骤8: 释放方块2...")
            self.open_gripper(0.06)
            
            # 9. 撤回
            print("步骤9: 撤回...")
            retreat_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.25
            ])
            if not self.move_to_pose(retreat_pose, wrist_rotation_angle):
                print("撤回失败")
                return False
            
            # === 第二阶段：将block3叠加到block2上 ===
            print("\n=== 阶段2：将方块3叠加到方块2上 ===")
            
            # 10. 移动到block3上方
            print("步骤10: 移动到方块3上方...")
            above_block3_pose = SimplePose([
                block3_pose.p[0],
                block3_pose.p[1], 
                block3_pose.p[2] + 0.15
            ])
            if not self.move_to_pose(above_block3_pose, wrist_rotation_angle):
                print("移动到方块3上方失败")
                return False
            
            # 11. 下降抓取block3
            print("步骤11: 下降抓取方块3...")
            grasp_block3_pose = SimplePose([
                block3_pose.p[0],
                block3_pose.p[1], 
                block3_pose.p[2] + 0.1
            ])
            if not self.move_to_pose(grasp_block3_pose, wrist_rotation_angle):
                print("下降到方块3失败")
                return False
            
            # 12. 夹取block3
            print("步骤12: 夹取方块3...")
            self.close_gripper()
            
            # 13. 提升block3
            print("步骤13: 提升方块3...")
            lift_block3_pose = SimplePose([
                block3_pose.p[0],
                block3_pose.p[1], 
                block3_pose.p[2] + 0.3
            ])
            if not self.move_to_pose(lift_block3_pose, wrist_rotation_angle):
                print("提升方块3失败")
                return False
            
            # 14. 移动到block2(现在在block1上)上方
            print("步骤14: 移动到叠加后的方块2上方...")
            above_stacked_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.30  # 两个方块的高度 + 安全距离
            ])
            if not self.move_to_pose(above_stacked_pose, wrist_rotation_angle):
                print("移动到叠加方块上方失败")
                return False
            
            # 15. 将block3放置到block2上
            print("步骤15: 将方块3放置到方块2上...")
            place_block3_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.25  # 三层方块的高度
            ])
            if not self.move_to_pose(place_block3_pose, wrist_rotation_angle):
                print("放置方块3失败")
                return False
            
            # 16. 释放block3
            print("步骤16: 释放方块3...")
            self.open_gripper(0.06)
            
            # # 17. 最终撤回
            # print("步骤17: 最终撤回...")
            # final_retreat_pose = SimplePose([
            #     block1_pose.p[0] + 0.1,
            #     block1_pose.p[1], 
            #     block1_pose.p[2] + 0.3
            # ])
            # if not self.move_to_pose(final_retreat_pose, wrist_rotation_angle):
            #     print("最终撤回失败")
            #     return False
            
            print("方块叠加任务完成!")
            return True
            
        except Exception as e:
            print(f"执行方块叠加时出错: {e}")
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
    
    epochs = 10
    for i in range(epochs):
        # 创建完整环境
        success_epochs = 0
        env.create_environment_with_planning(robot_config)
        env.debug_positions()
        try:
            success = env.plan_block_stacking()
            if success:
                print("任务执行成功!")
                success_epochs+=1
            else:
                print("任务执行失败!")
                
            # 继续运行仿真以观察结果
            print("继续运行仿真...")
            for i in range(200):
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
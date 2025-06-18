
from .base_task import Base_task
from .utils import *
import sapien
import math

class blocks_stack_hard(Base_task):

    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot(**kwags)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()
        self.pre_move()
        self.robot.set_origin_endpose()
        self.load_actors()
        
        self.step_lim = 850

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
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
            qpos=[1,0,0,0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0,0,1.],
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[1,0,0,0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,0,1.],
            )

        self.block1 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(1,0,0),
            name="box"
        )

        block_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.76],
            qpos=[1,0,0,0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0,0,1.],
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - self.block1.get_pose().p[:2],2)) < 0.01 \
              or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[1,0,0,0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,0,1.],
            )


        self.block2 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(0,1,0),
            name="box"
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - self.block1.get_pose().p[:2],2)) < 0.01 or \
              np.sum(pow(block_pose.p[:2] - self.block2.get_pose().p[:2],2)) < 0.01 or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[1,0,0,0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,0,1.],
            )


        self.block3 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(0,0,1),
            name="box"
        )

        self.block1_data = self.block2_data  = self.block3_data = create_block_data([0.025,0.025,0.025])

        self.block1.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.block2.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.block3.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        pose = self.block1.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04])
        pose = self.block2.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04]) 
        pose = self.block3.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04])
        # target_pose = [-0.04,-0.13,0.04,-0.05]
    #     self.prohibited_area.append(target_pose)

    def play_once(self):
        # Retrieve actor objects and data
        self.las_gripper = None
        block1_target_pose = [0, -0.13, 0.75]
        # Pick up block1 and move it to the target position
        self.pick_and_place_block(self.block1, self.block1_data, block1_target_pose)
        self.pick_and_place_block(self.block2, self.block2_data, self.block1, self.block1_data)
        self.pick_and_place_block(self.block3, self.block3_data, self.block2, self.block2_data)

        info = dict()
        info['messy_table_info'] = self.record_messy_objects
        info['texture_info'] = {'wall_texture': self.wall_texture, 'table_texture': self.table_texture}
        return info
        
    def pick_and_place_block(self, block, block_data, target_block, target_block_data = None):
        # Get the pre-grasp and target grasp poses
        block_pose = block.get_pose().p
        arm_tag = 'left' if block_pose[0] < 0 else 'right'
        pre_grasp_pose = self.get_grasp_pose_w_labeled_direction(block, block_data, pre_dis=0.09, contact_point_id=0)
        target_grasp_pose = self.get_grasp_pose_w_labeled_direction(block, block_data, pre_dis=0, contact_point_id=0)
        
        move_func = self.left_move_to_pose if arm_tag == 'left' else self.right_move_to_pose
        close_gripper = self.close_left_gripper if arm_tag == 'left' else self.close_right_gripper
        open_gripper = self.open_left_gripper if arm_tag == 'left' else self.open_right_gripper
        
        if self.las_gripper is not None and self.las_gripper != arm_tag:
            left_target_pose = self.robot.left_original_pose if self.las_gripper == 'left' else pre_grasp_pose
            right_target_pose = self.robot.right_original_pose if self.las_gripper == 'right' else pre_grasp_pose
            self.together_move_to_pose(left_target_pose=left_target_pose, right_target_pose=right_target_pose)
        else:
            move_func(pre_grasp_pose)

        # move_func(pre_grasp_pose)
        move_func(target_grasp_pose)
        close_gripper()
        move_func(pre_grasp_pose)
        
        target_pose = self.get_actor_goal_pose(target_block, target_block_data, 0)
        target_approach_direction = [0,1,0,0]
        pre_place_pose = self.get_grasp_pose_from_goal_point_and_direction(block, block_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=target_approach_direction, pre_dis=0.09)
        target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(block, block_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=target_approach_direction, pre_dis=0)
        
        move_func(pre_place_pose)
        move_func(target_place_pose)
        open_gripper()
        move_func(pre_place_pose)
        self.las_gripper = arm_tag

    def get_target_grap_pose(self,shoe_rpy):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        target_pose = [0,-0.13]
        eps = [0.025,0.025,0.01]
        reward = 0
        if np.all(abs(block1_pose[:2] - np.array(target_pose)) < eps[:2]) and \
           np.all(abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2]+0.05])) < eps) and self.is_left_gripper_open() and self.is_right_gripper_open():
            reward += 0.4
            if np.all(abs(block3_pose - np.array(block2_pose[:2].tolist() + [block2_pose[2]+0.05])) < eps) and self.is_left_gripper_open() and self.is_right_gripper_open():
                reward += 0.6
        return reward

    def stage_reward(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        target_pose = [0,-0.13]
        eps = [0.025,0.025,0.01]
        reward = 0
        if self.is_right_gripper_open() and np.all(abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2]+0.05])) < eps):
            reward += 0.3
            if np.all(abs(block3_pose - np.array(block2_pose[:2].tolist() + [block2_pose[2]+0.05])) < eps) and self.is_left_gripper_open() and self.is_right_gripper_open():
                reward += 0.7
        return reward

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        target_pose = [0,-0.13]
        eps = [0.025,0.025,0.01]
        return np.all(abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2]+0.05])) < eps) and \
               np.all(abs(block3_pose - np.array(block2_pose[:2].tolist() + [block2_pose[2]+0.05])) < eps) and self.is_left_gripper_open() and self.is_right_gripper_open()
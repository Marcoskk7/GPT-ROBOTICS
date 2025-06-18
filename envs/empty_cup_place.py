from .base_task import Base_task
from .utils import *
import sapien

class empty_cup_place(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot(**kwags)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()
        self.pre_move()
        self.robot.set_origin_endpose()
        self.load_actors()
        
        self.step_lim = 500
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        tag = np.random.randint(0,2)
        if tag==0:
            self.cup,self.cup_data = rand_create_glb(
                self.scene,
                xlim=[0.15,0.3],
                ylim=[-0.2,0.05],
                zlim=[0.8],
                modelname="022_cup",
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )
            cup_pose = self.cup.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.05,0.1],
                ylim=[-0.2,0.05],
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )

            while np.sum(pow(cup_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.05,0.1],
                    ylim=[-0.2,0.05],
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
                ylim=[-0.2,0.05],
                zlim=[0.8],
                modelname="022_cup",
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )
            cup_pose = self.cup.get_pose().p

            coaster_pose = rand_pose(
                xlim=[-0.1, 0.05],
                ylim=[-0.2,0.05],
                zlim=[0.76],
                rotate_rand=False,
                qpos=[0.5,0.5,0.5,0.5],
            )

            while np.sum(pow(cup_pose[:2] - coaster_pose.p[:2],2)) < 0.01:
                coaster_pose = rand_pose(
                    xlim=[-0.1, 0.05],
                    ylim=[-0.2,0.05],
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

    def play_once(self):
        # Get the current pose of the cup
        cup_pose = self.get_actor_functional_pose(self.cup, self.cup_data)

        # Determine which arm to use based on the cup's x coordinate
        if cup_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose
            close_gripper_function = self.close_right_gripper
            open_gripper_function = self.open_right_gripper
            contant_id = 0
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose
            close_gripper_function = self.close_left_gripper
            open_gripper_function = self.open_left_gripper
            contant_id = 2

        # Get the pre-grasp and target grasp poses for the cup
        pre_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.cup, actor_data=self.cup_data, pre_dis=0.1, contact_point_id = contant_id)
        target_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.cup, actor_data=self.cup_data, pre_dis=0., contact_point_id = contant_id)

        # Move to the pre-grasp pose
        move_function(pre_grasp_pose)
        open_gripper_function(0.85)

        # Move to the target grasp pose and close the gripper to pick up the cup
        move_function(target_grasp_pose)
        close_gripper_function()  # Tighten the gripper to ensure a secure grasp

        # Lift the cup slightly
        lift_pose = target_grasp_pose.copy()
        lift_pose[2] += 0.1  # Lift the cup by 0.1 meters
        move_function(lift_pose)

        # Get the target pose for placing the cup on the coaster
        coaster_pose = self.get_actor_goal_pose(self.coaster, self.coaster_data, id=0)
        place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor=self.cup, actor_data=self.cup_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=coaster_pose, target_approach_direction=[0,0.707,0.707,0], pre_dis=0.05)
        target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor=self.cup, actor_data=self.cup_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=coaster_pose, target_approach_direction=[0,0.707,0.707,0], pre_dis=0)

        # Move to the pre-place pose
        move_function(place_pose)

        # Move to the target place pose and open the gripper to place the cup
        move_function(target_place_pose)
        open_gripper_function(0.85)

        # Lift the arm slightly after placing the cup
        lift_pose = target_place_pose.copy()
        lift_pose[2] += 0.1  # Lift the arm by 0.1 meters
        move_function(lift_pose)

        info = dict()
        info['messy_table_info'] = self.record_messy_objects
        info['texture_info'] = {'wall_texture': self.wall_texture, 'table_texture': self.table_texture}
        return info
    
    def stage_reward(self):
        eps = 0.025
        coaster_pose = self.coaster.get_pose().p
        cup_pose = self.cup.get_pose().p
        if abs(cup_pose[0] - coaster_pose[0])<eps  and  abs(cup_pose[1] - coaster_pose[1])<eps and (cup_pose[2] - 0.792) < 0.005:
            return 1
        return 0

    def check_success(self):
        eps = 0.025
        coaster_pose = self.coaster.get_pose().p
        cup_pose = self.cup.get_pose().p
        return abs(cup_pose[0] - coaster_pose[0])<eps  and  abs(cup_pose[1] - coaster_pose[1])<eps and (cup_pose[2] - 0.792) < 0.005
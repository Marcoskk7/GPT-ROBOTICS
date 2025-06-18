import mplib.planner
import mplib
import numpy as np
import pdb
import numpy as np
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld

class MplibPlanner():
    # links=None, joints=None
    def __init__(self, urdf_path, srdf_path, move_group, robot_origion_pose, robot_entity, planner_type = 'mplib_RRT', scene = None):
        super().__init__()
        ta.setup_logging("CRITICAL") # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf = urdf_path,
                srdf = srdf_path,
                move_group = move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False
            )

            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(
                planning_world,
                move_group,
                # user_link_names=links,
                # user_joint_names=joints,
                # use_convex=False
            )
            # self.planner.planning_world.get_allowed_collision_matrix().set_entry(
            #     "panda_link0", "ground_1", True
            # )

        self.planner_type = planner_type

        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP
    
    def show_info(self):
        print('joint_limits', self.planner.joint_limits)
        print('joint_acc_limits', self.planner.joint_acc_limits)

    def plan_pose(self, now_qpos, target_pose, use_point_cloud=False, use_attach=False, arms_tag = None, try_times=2, log = True):
        result = {}
        result['status'] = 'Fail'

        now_try_times = 1
        while result['status'] != 'Success' and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1/250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm palnning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm palnning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(self, now_qpos, target_pose, use_point_cloud=False, use_attach=False, arms_tag = None, log = False):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )
        
        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm palnning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm palnning wrong! (step = {n_step})")
                result["status"] = "Fail"
        
        return result
    
    def plan_path(self, now_qpos, target_pose, use_point_cloud=False, use_attach=False, arms_tag = None, log = True):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == 'mplib_RRT':
            result = self.plan_pose(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, try_times=10, log = log)
        elif self.planner_type == 'mplib_screw':
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)
        
        return result
    
    def plan_grippers(self, now_val, target_val):
        step_n = 200
        dis_val = target_val - now_val
        step = dis_val / step_n
        res={}
        vals = np.linspace(now_val, target_val, step_n)
        res['step_n'] = step_n
        res['step'] = step
        res['result'] = vals
        return res
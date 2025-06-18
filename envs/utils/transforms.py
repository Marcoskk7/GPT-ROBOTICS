import cv2
import scipy
import sapien
import numpy as np
import transforms3d as t3d
from scipy.ndimage import gaussian_filter

def transform_pts(pts:np.ndarray, RT:np.ndarray):
    '''
        将输入的点坐标列表分别施加输入的变换

        Args:
            pts: np.ndarray, 点坐标列表
            RT: np.ndarray, 变换矩阵
    '''
    n = pts.shape[0]
    pts = np.concatenate([pts, np.ones((n, 1))], axis=1)
    pts = RT @ pts.T
    pts = pts.T[:, :3]
    return pts

def estimate_rigid_transform(P:np.ndarray, Q:np.ndarray):
    assert P.shape == Q.shape
    n, dim = P.shape
    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)
    C = np.dot(np.transpose(centeredP), centeredQ) / n

    try:
        V, S, W = np.linalg.svd(C)
        d = np.linalg.det(V) * np.linalg.det(W)
        D = np.eye(3)
        D[2, 2] = d
        R = np.dot(np.dot(V, D), W)
    except Exception as e:
        print(e)
        try:
            V, S, W = scipy.linalg.svd(C, lapack_driver='gesvd')
            d = np.linalg.det(V) * np.linalg.det(W)
            D = np.eye(3)
            D[2, 2] = d
            R = np.dot(np.dot(V, D), W)
        except Exception as e2:
            print(e2)
            R = np.eye(3)

    t = Q.mean(axis=0) - P.mean(axis=0).dot(R)

    return R, t

def quat_product(q1:np.ndarray, q2:np.ndarray):
    '''
        计算两个四元数的乘积，用于计算旋转

        Args:
            q1, q2: np.ndarray, 两个四元数
    '''
    r1 = q1[0]
    r2 = q2[0]
    v1 = np.array([q1[1], q1[2], q1[3]])
    v2 = np.array([q2[1], q2[2], q2[3]])
    r = r1 * r2 - np.dot(v1, v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([r, v[0], v[1], v[2]])

    return q

def cv2ex2pose(ex):
    cv2sapien = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]], dtype=np.float32)

    pose = ex @ np.linalg.inv(cv2sapien)

    return sapien.Pose(pose)

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K` dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0

def generate_patch_array(super_resolution_ratio=10):
    circle_radius = 3
    size_slot_num = 50
    base_circle_radius = 1.5

    patch_array = np.zeros(
        (super_resolution_ratio, super_resolution_ratio, size_slot_num, 4 * circle_radius, 4 * circle_radius),
        dtype=np.uint8)
    for u in range(super_resolution_ratio):
        for v in range(super_resolution_ratio):
            for w in range(size_slot_num):
                img_highres = np.ones(
                    (4 * circle_radius * super_resolution_ratio, 4 * circle_radius * super_resolution_ratio),
                    dtype=np.uint8) * 255
                center = np.array(
                    [circle_radius * super_resolution_ratio * 2, circle_radius * super_resolution_ratio * 2],
                    dtype=np.uint8)
                center_offseted = center + np.array([u, v])
                radius = round(base_circle_radius * super_resolution_ratio + w)
                img_highres = cv2.circle(img_highres, tuple(center_offseted), radius, (0, 0, 0), thickness=cv2.FILLED,
                                         lineType=cv2.LINE_AA)
                img_highres = cv2.GaussianBlur(img_highres, (17, 17), 15)
                img_lowres = cv2.resize(img_highres, (4 * circle_radius, 4 * circle_radius),
                                        interpolation=cv2.INTER_CUBIC)
                patch_array[u, v, w, ...] = img_lowres

    return {
        "base_circle_radius": base_circle_radius,
        "circle_radius": circle_radius,
        "size_slot_num": size_slot_num,
        "patch_array": patch_array,
        "super_resolution_ratio": super_resolution_ratio,
    }

class Point:
    points: list['Point'] = []

    '''特定 base 坐标系下的点'''
    def __init__(self,
                scene:sapien.Scene,
                base:sapien.Entity,
                base_scale: float,
                init_mat:np.ndarray,
                base_pose_mat:np.ndarray=None,
                scaled:bool=True,
                follow:sapien.Entity=None,
                name:str='point',
                size:float=0.05,
                eular_round_to:int=0.01):
        self.name = name
        self.scene = scene
        self.base = base
        if base_pose_mat is not None:
            self.base_pose_mat = np.array(base_pose_mat)
        else:
            self.base_pose_mat = base.get_pose().to_transformation_matrix()
        self.follow = follow
        self.base_scale = base_scale
        self.eular_round_to = eular_round_to
        
        self.mat = np.array(init_mat)
        if not scaled:
            self.mat[:3, 3] *= self.base_scale
        
        self.pose = self.trans_base(self.base.get_pose().to_transformation_matrix(), self.base_pose_mat, self.mat)
        self.mat = self.word2base(self.pose.to_transformation_matrix()).to_transformation_matrix()
        self.base_pose_mat = self.base.get_pose().to_transformation_matrix()

        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_visual_from_file(
            filename='./assets/objects/cube/textured.obj',
            scale=[size, size, size]
        )
        self.point = builder.build(name=name)
        self.point.set_pose(self.pose)
        Point.points.append(self)
    
    def __del__(self):
        Point.points.remove(self)
    
    def get_pose(self) -> sapien.Pose:
        return self.pose
    
    @staticmethod
    def pose2list(pose:sapien.Pose) -> list:
        return pose.p.tolist() + pose.q.tolist()
    
    @staticmethod
    def round_eular(eular, round_to:int=1) -> np.ndarray:
        unit = round_to/180*np.pi
        return np.round(np.array(eular)/unit)*unit
    
    @staticmethod
    def trans_mat(to_mat:np.ndarray, from_mat:np.ndarray, scale:float=1.):
        to_rot = to_mat[:3, :3]
        from_rot = from_mat[:3, :3]
        rot_mat = to_rot @ from_rot.T

        trans_mat = (to_mat[:3, 3] - from_mat[:3, 3])/scale

        result = np.eye(4)
        result[:3, :3] = rot_mat
        result[:3, 3] = trans_mat
        result = np.where(np.abs(result) < 1e-5, 0, result)
        return result

    @staticmethod
    def trans_pose(to_pose:sapien.Pose, from_pose:sapien.Pose, scale:float=1.):
        return Point.trans_mat(
            to_pose.to_transformation_matrix(),
            from_pose.to_transformation_matrix(),
            scale
        )
    
    @staticmethod
    def trans_base(now_base_mat:np.ndarray, init_base_mat:np.ndarray, init_pose_mat:np.ndarray, scale:float=1.):
        now_base_mat = np.array(now_base_mat)
        init_base_mat = np.array(init_base_mat)
        init_pose_mat = np.array(init_pose_mat)
        init_pose_mat[:3, 3] *= scale

        now_pose_mat = np.eye(4)
        base_trans_mat = Point.trans_mat(
            now_base_mat,
            init_base_mat
        )
        now_pose_mat[:3, :3] = base_trans_mat[:3,:3] @ init_pose_mat[:3, :3] @ base_trans_mat[:3, :3].T
        now_pose_mat[:3, 3] = base_trans_mat[:3, :3] @ init_pose_mat[:3, 3]

        # 转化为世界坐标
        p = now_pose_mat[:3, 3] + now_base_mat[:3, 3]
        q_mat = now_pose_mat[:3, :3] @ now_base_mat[:3, :3]
        return sapien.Pose(p, t3d.quaternions.mat2quat(q_mat))

    def get_output_mat(self):
        opt_mat = self.mat.copy()
        opt_mat[:3, 3] /= self.base_scale
        return opt_mat

    def base2world(self, entity_mat, scale=1.) -> sapien.Pose:
        '''将 base 坐标系下的矩阵转换到世界坐标系下'''
        entity_mat = np.array(entity_mat)
        base_mat = self.base.get_pose().to_transformation_matrix()
        p = entity_mat[:3, 3]*scale + base_mat[:3, 3]
        q_mat = entity_mat[:3, :3] @ base_mat[:3, :3]
        return sapien.Pose(p, t3d.quaternions.mat2quat(q_mat))

    def word2base(self, entity_mat, scale=1.) -> sapien.Pose:
        '''将世界坐标系下的矩阵转换到 base 坐标系下'''
        entity_mat = np.array(entity_mat)
        base_mat = self.base.get_pose().to_transformation_matrix()
        p = entity_mat[:3, 3] - base_mat[:3, 3]
        q_mat = entity_mat[:3, :3] @ base_mat[:3, :3].T
        return sapien.Pose(p, t3d.quaternions.mat2quat(q_mat))
    
    def set_pose(self, new_pose:sapien.Pose):
        '''更新点的位置'''
        self.pose = new_pose
        self.point.set_pose(self.pose)
        self.mat = self.word2base(new_pose.to_transformation_matrix()).to_transformation_matrix()

        print('set', self.name)
        print(self.get_output_mat().tolist())
        print()

    def update(self, force_output:bool=False, flexible:bool=False):
        new_mat = np.eye(4)
        if self.follow is not None:
            new_mat = self.trans_mat(
                self.follow.get_pose().to_transformation_matrix(),
                self.base.get_pose().to_transformation_matrix()
            )
        elif flexible:
            new_mat = self.trans_mat(
                self.point.get_pose().to_transformation_matrix(),
                self.base.get_pose().to_transformation_matrix()
            )
        else:
            new_mat = self.word2base(
                self.trans_base(
                    self.base.get_pose().to_transformation_matrix(), 
                    self.base_pose_mat, 
                    self.mat
                ).to_transformation_matrix()
            ).to_transformation_matrix()

        new_mat[:3, :3] = t3d.euler.euler2mat(
            *self.round_eular(t3d.euler.mat2euler(new_mat[:3, :3]), self.eular_round_to))
        self.pose = self.base2world(new_mat)
        self.point.set_pose(self.pose)
        
        if not np.allclose(new_mat, self.mat, atol=1e-3) or force_output:
            self.mat = new_mat
            self.base_pose_mat = self.base.get_pose().to_transformation_matrix()
            print('update', self.name)
            if self.name == 'left':
                print('\'lb\': ', self.get_output_mat().tolist(), ', ', sep='')
            elif self.name == 'right':
                print('\'rb\': ', self.get_output_mat().tolist(), ', ', sep='')
            else:
                print(self.get_output_mat().tolist())
            print('init_base_mat =', self.base.get_pose().to_transformation_matrix().tolist())
            print()
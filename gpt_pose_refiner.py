import numpy as np
import cv2
from scipy.optimize import least_squares

class PoseRefiner:
    def __init__(self, camera_matrix, dist_coeffs):
        self.K = camera_matrix
        self.D = dist_coeffs

    def refine_pose(self, obj_pts, img_pts, rvec_init, tvec_init):
        def reprojection_error(params):
            rvec = params[:3].reshape((3, 1))
            tvec = params[3:].reshape((3, 1))
            projected_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
            projected_pts = projected_pts.reshape(-1, 2)
            return (projected_pts - img_pts).ravel()

        init_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        result = least_squares(reprojection_error, init_params, method='lm')

        refined_rvec = result.x[:3].reshape((3, 1))
        refined_tvec = result.x[3:].reshape((3, 1))
        return refined_rvec, refined_tvec, result.cost
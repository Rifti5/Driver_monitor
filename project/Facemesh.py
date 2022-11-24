import mediapipe as mp
import numpy
import cv2

from misc import get_euclidian_distance


class FaceMesh:
    def __init__(self, max_faces: int,
                 min_detection_confidence: float,
                 min_tracking_confidence: float,
                 blink_threshold: float):

        self.__mp_face_mesh = mp.solutions.face_mesh
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles

        self.__mesh = self.__mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_landmarks=True
        )

        self.__points_3d = numpy.array([
                            (0.5, 0.5, 0.0),
                            (0.5, 1.0, -3.0),
                            (1.0, 0.4, -5.0),
                            (0.01, 0.4, -5.0),
                            (1.0, 0.2, -4.0),
                            (0.01, 0.2, -4.0)
                        ])

        self.results = None
        self.world_coordinates = None

        self.is_blinked = [False, False]
        self.blink_threshold = blink_threshold
        self.irises_pos_w = None

        self.translation_vector = None
        self.rotation_vector = None

    def process(self, frame: numpy.array, frame_w: int, frame_h: int):
        results = self.__mesh.process(frame)
        if results.multi_face_landmarks:
            self.results = results
            self.world_coordinates = numpy.array([(int(landmark.x * frame_w), int(landmark.y * frame_h)) for landmark in results.multi_face_landmarks[0].landmark])
            return True
        else:
            self.results = None
            return False

    def draw_face_mesh(self, frame: numpy.array,
                       draw_tesselation: bool = False,
                       draw_contour: bool = True,
                       draw_irises: bool = True):
        for face_landmarks in self.results.multi_face_landmarks:
            if draw_tesselation:
                self.__mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.__mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.__mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            if draw_contour:
                self.__mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.__mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.__mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            if draw_irises:
                self.__mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.__mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.__mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

    def detect_blink(self):

        r_v = get_euclidian_distance(self.world_coordinates[257], self.world_coordinates[253])
        r_h = get_euclidian_distance(self.world_coordinates[463], self.world_coordinates[359])

        l_v = get_euclidian_distance(self.world_coordinates[27], self.world_coordinates[23])
        l_h = get_euclidian_distance(self.world_coordinates[130], self.world_coordinates[243])

        r_ratio = abs(r_v/r_h)
        l_ratio = abs(l_v/l_h)

        l_blink = True if l_ratio <= self.blink_threshold else False
        r_blink = True if r_ratio <= self.blink_threshold else False

        self.is_blinked = [l_blink, r_blink]

    def calc_face_rotation(self, frame, camera_matrix: numpy.array):
        points_2d = numpy.array([self.world_coordinates[1],
                               self.world_coordinates[152],
                               self.world_coordinates[454],
                               self.world_coordinates[234],
                               self.world_coordinates[251],
                               self.world_coordinates[21]],
                               dtype='float')

        success, rotation_vector, translation_vector = cv2.solvePnP(self.__points_3d, points_2d, camera_matrix,
                                                                    None, cv2.SOLVEPNP_EPNP, useExtrinsicGuess=False)

        self.translation_vector = translation_vector
        self.rotation_vector = rotation_vector

    def draw_rotation_vector(self, frame, camera_matrix):
        nose_end_point2D, jacobian = cv2.projectPoints(numpy.array([(0.0, 0.0, 1000.0)]), self.rotation_vector,
                                                       self.translation_vector, camera_matrix, None)

        point1 = (int(self.world_coordinates[1][0]), int(self.world_coordinates[1][1]))
        point2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, point1, point2, (0, 255, 255), 2)



    def calc_iris_position(self):
        irises_coordinates_w = numpy.array([(self.world_coordinates[474],
                                            self.world_coordinates[475],
                                            self.world_coordinates[476],
                                            self.world_coordinates[477]),
                                           (self.world_coordinates[469],
                                            self.world_coordinates[470],
                                            self.world_coordinates[471],
                                            self.world_coordinates[472])])

        irises_centres_w = numpy.array([(sum(p[0] for p in irises_coordinates_w[0]) / 4,
                                        sum(p[0] for p in irises_coordinates_w[0]) / 4),
                                        (sum(p[0] for p in irises_coordinates_w[1]) / 4,
                                         sum(p[0] for p in irises_coordinates_w[1]) / 4)
                                        ])

        self.irises_pos_w = irises_centres_w

    #     eye_coordinates_w = numpy.array([(self.world_coordinates[386],
    #                                       self.world_coordinates[359],
    #                                       self.world_coordinates[463],
    #                                       self.world_coordinates[253]),
    #                                      (self.world_coordinates[159],
    #                                       self.world_coordinates[243],
    #                                       self.world_coordinates[23],
    #                                       self.world_coordinates[130])])
    #
    #     bbox_sizes = numpy.array([(abs(eye_coordinates_w[0][1][0]-eye_coordinates_w[0][3][0]),
    #                               abs(eye_coordinates_w[0][0][1]-eye_coordinates_w[0][2][1])),
    #                              (abs(eye_coordinates_w[1][1][0] - eye_coordinates_w[1][3][0]),
    #                               abs(eye_coordinates_w[1][0][1] - eye_coordinates_w[1][2][1]))])
    #
    #     bbox_origins = numpy.array([(eye_coordinates_w[0][3][0], eye_coordinates_w[0][0][1]),
    #                                 (eye_coordinates_w[1][3][0], eye_coordinates_w[1][0][1])])
    #
    #     x, y = self.__normalize_world_coordinates_inside_bbox(irises_centres_w[0], bbox_sizes[0], bbox_origins[0])
    #     print(x, y)
    #
    # @staticmethod
    # def __normalize_world_coordinates_inside_bbox(point, bbox_size, bbox_origin):
    #     # print(bbox_origin)
    #     x = abs((point[0] - bbox_origin[0])) / bbox_size[0]
    #     y = abs((point[1] - bbox_origin[1])) / bbox_size[1]
    #     return x, y

import cv2


from Camera import Camera
from Facemesh import FaceMesh
from SleepDetector import SleepDetector
from Logger import Logger
from misc import *
from AlarmThread import AlarmThread


def main():
    cam = Camera(0)
    width, height = cam.get_frame_shape()
    camera_matrix = numpy.array([[height, 0., height / 2],
                                [0., height, width / 2],
                                [0., 0., 1.]])

    mesh = FaceMesh(1, 0.5, 0.5, 0.57)
    sleep_detector = SleepDetector(2)

    logger = Logger('D:\\driver monitor\\Logs\\')
    logger.create_log_file()

    alarm_thread = AlarmThread()
    alarm_thread.pause()
    alarm_thread.start()

    while cam.is_opened():
        success, frame = cam.get_frame()

        if not success:
            continue

        set_frame_writable(frame, False)
        frame = convert_color_bgr_to_rgb(frame)

        mesh_success = mesh.process(frame, width, height)

        set_frame_writable(frame, True)
        frame = convert_color_rgb_to_bgr(frame)

        if mesh_success:
            mesh.detect_blink()

            sleep_detector.detect(mesh.is_blinked)

            if all(mesh.is_blinked):
                logger.write_log('0')

            if sleep_detector.alarm:
                alarm_thread.resume()
            else:
                alarm_thread.pause()

            mesh.calc_face_rotation(frame, camera_matrix)
            mesh.calc_iris_position()

            roll, pitch, yaw = vec3_rad_to_deg(mesh.rotation_vector)

            mesh.draw_face_mesh(frame)
            mesh.draw_rotation_vector(frame, camera_matrix)

            cv2.rectangle(frame, (0, 0), (180, 260), (0, 0, 0), -1)

            draw_text(frame, f'roll:{round(roll, 2)}', (20, 20), color=(0, 255, 255))
            draw_text(frame, f'pitch:{round(pitch, 2)}', (20, 40), color=(0, 255, 255))
            draw_text(frame, f'yaw:{round(yaw, 2)}', (20, 60), color=(0, 255, 255))

            draw_text(frame, f'R eye closed:{mesh.is_blinked[0]}', (20, 100), color=(0, 0, 255))
            draw_text(frame, f'L eye closed:{mesh.is_blinked[1]}', (20, 120), color=(0, 255, 0))

            draw_text(frame, f'R iris pos:{round(mesh.irises_pos_w[0][0])} {round(mesh.irises_pos_w[0][1])}',
                      (20, 140), color=(0, 0, 255))
            draw_text(frame, f'L iris pos:{round(mesh.irises_pos_w[1][0])} {round(mesh.irises_pos_w[1][1])}',
                      (20, 160), color=(0, 255, 0))

            draw_text(frame, f'Alarm Delay:{sleep_detector.alarm_delay}', (20, 200))
            draw_text(frame, f'Countdown:{sleep_detector.get_countdown()}', (20, 220))
            draw_text(frame, f'Alarm:{sleep_detector.alarm}', (20, 240))

        else:
            draw_text(frame, f'Face not found', (20, 20), color=(255, 0, 255))

        cv2.imshow('DriverMonitor', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            alarm_thread.stop()
            break


if __name__ == '__main__':
    main()
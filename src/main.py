import os
import CheckCamera
import CaptureImage
import LiveRecognition
import Recognition 
import TrainImage


def title_bar():
    os.system('cls')  # for windows

    # title of the program

    print("\t**********************************************")
    print("\t***** Face Recognition Attendance System *****")
    print("\t**********************************************")


def mainMenu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Kiem tra Camera")
    print("[2] Ghi danh")
    print("[3] Huan luyen du lieu")
    print("[4] Nhan dang nguoi trong anh")
    print("[5] Diem danh")
    print("[6] Quit")

    while True:
        try:
            i = int(input("Nhap tu ban phim:"))
            # print(i)
            if i == 1:
                CheckCamera.camera()
                return mainMenu()
            elif i == 2:
                CaptureImage.takeImages()
                return mainMenu()
            elif i == 3:
                TrainImage.FaceClassification()
                return mainMenu()
            elif i == 4:
                LiveRecognition.liveRecognition()
                return mainMenu()
            elif i == 5:
                Recognition.recognizeAttendance()
                return mainMenu()
            elif i==6:
                print("thoat")
                exit()
            else:
                print("invald")
                mainMenu()
        except ValueError:
            print("Khong hop le")

mainMenu()
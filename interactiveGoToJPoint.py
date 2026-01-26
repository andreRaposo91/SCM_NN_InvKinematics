import time
import serial # pip install pyserial

robot = serial.Serial()
robot.port = "COM5" # utilizar "python -m serial.tools.list_ports" para encontror porta
robot.baudrate = 9600
robot.open()

time.sleep(2)

while robot.in_waiting > 1:
    print(robot.readline())

time.sleep(2)

# robot.write(b"100, 40, 65\n")
while robot.in_waiting > 1:
    print(robot.readline())

time.sleep(2)

while True:
    pos = input("Next position (enter to quit):")
    # pos = '40,40,40'
    if pos == "": break
    else: pos += "\n"
    robot.write(pos.encode())
    time.sleep(1.5)
    while robot.in_waiting > 1:
        print(robot.readline())

while robot.in_waiting > 1:
    print(robot.readline())
time.sleep(1)
print("Homing")
# robot.write(b"65, 65, 65\n")
robot.write(b"1220, 1220, 1220\n")
while robot.in_waiting > 1:
    print(robot.readline())
time.sleep(1)

robot.close()
print("Port released. Goodbye!")

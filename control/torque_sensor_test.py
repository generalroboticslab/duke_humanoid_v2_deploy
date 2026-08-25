"""Bench script: runs its routine only when invoked (python torque_sensor_test.py);
importing it is inert; it has no CLI flags. main() opens /dev/ttyACM0
(115200 8N1, hard-coded) and publishes every 8-byte ASCII torque reading as
{"torque": float} over UDP to localhost:9870 (the telemetry port) until
Ctrl+C. Torque-sensor bench one-off.
"""

import serial
from ipc.publisher import DataPublisher


def main():
    # 08-22: moved under main() so that importing the module is inert.
    publisher = DataPublisher(
        target_url="udp://localhost:9870", encoding="msgpack",broadcast=False,thread=True)


    ser = serial.Serial(
        port='/dev/ttyACM0',  # Replace with the correct port
        baudrate=115200,  # Replace with the actual baud rate
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE
    )
    ser.set_low_latency_mode(True)

    try:
        while True:
            # ascii mode
            data = ser.read(8)
            data = float(data.decode('ascii').strip())
            # print(f"{data:+5.2f}")
            publisher.publish({"torque": data})


            # # Read data from the serial port
            # data = ser.read(6)  # Read 6 bytes of data
            # # Check if enough data is received
            # if len(data) == 6:
            #     # Extract torque and speed values
            #     # torque = int.from_bytes(data[0:2], byteorder='big')/65536
            #     torque = float((data[1] << 8) | data[0])

            #     # Extract speed value and torque sign
            #     if data[2]>=128: # sign bit
            #         torque_sign = -1
            #         torque = -torque
            #     else:
            #         torque_sign = 1
            #     print(f"{torque:+5.2f}")

            #     # speed = speed_bytes & 0x7FFF  # Mask out the sign bit

            #     # # Apply the sign to the torque value
            #     # torque *= torque_sign

            #     # # Print the values
            #     # print(f"Torque: {torque}, Speed: {speed}")

    except KeyboardInterrupt:
        # Close the serial port on keyboard interrupt
        ser.close()


if __name__ == "__main__":
    main()

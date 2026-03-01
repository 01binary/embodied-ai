# Command message for club actuator demo
# led/solenoid: desired output state
# servo_cmd: normalized target position in range [0.0, 1.0]
# servo_enable: if false, ignore servo_cmd and keep current target

bool led
bool solenoid
bool servo_enable
float32 servo_cmd

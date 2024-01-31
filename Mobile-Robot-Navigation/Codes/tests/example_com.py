from data import *
from communication import *
from tdmclient import ClientAsync, aw 
import time 

thymio = Thymio()

thymio.set_variable(Motors(50,50))

time.sleep(1)
thymio.set_variable(Lights([32,0,0]))


thymio.read_variables()
print("sensors : ", thymio.sensors)
print("motors : ", thymio.motors)
print("leds : ", thymio.leds)

thymio.stop()
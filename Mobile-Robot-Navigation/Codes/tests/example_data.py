from data import * 

A = Point(0, 0)
B = Point(2,0)
C = Point(0,2)

print(B)
print(A.dist(B))

ABC = Obstacle([A,B,C])
print(ABC)

map = Map([A, Point(10,0), Point(10,10), Point(0,10)],[ABC])
print(map)

thymio = Robot(A, 90)
print(thymio)

world = Environment(thymio, map, Point(7,8))
print(world)

motors = Motors(30,30)
#sensors = Sensors(readprox, readground)
#leds = Lights([r,g,b])

print(motors.__class__ is Motors) 

tes2 = thymio.copy()
tes2.position.x = 54
print(tes2, thymio)
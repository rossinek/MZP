import numpy as np
import cv2
import time

def euler_method(f, t, y, h):
    return y + h * f(t, y)

def midpoint_method(f, t, y, h):
    K1 = f(t, y)
    K2 = f(t+h*0.5, y+(h*0.5*K1))
    return y + h * K2

def RK4_method(f, t, y, h):
    K1 = f(t, y)
    K2 = f(t+h*0.5, y+(h*0.5*K1))
    K3 = f(t+h*0.5, y+(h*0.5*K2))
    K4 = f(t+h, y+(h*K3))
    return y + h*(K1/6+K2/3+K3/3+K4/6)

def mass_to_radius(m, min_r, max_r):
    if(np.max(m) == np.min(m)):
        r = np.ones_like(m)*max_r
    else:
        r = (m-np.min(m))*((max_r-min_r)/(np.max(m)-np.min(m)))+min_r
    return r.astype(int)

def y_to_pos(y,l,shape,pos_start):
    pos = []
    pos.append(pos_start)
    for i in range(len(y)):
        theta = y[i, 0]
        pos.append((
                pos[i][0]+int(l[i]*np.sin(theta)),
                pos[i][1]+int(l[i]*np.cos(theta))
            ))
    return pos

def render_frame(y, l, r, size, meter_scale=1):
    shape = (size,size,3)
    img = np.zeros(shape)
    
    y_b = y[1:]
        
    pos_block = (shape[0]//2+int(y[0,0]*meter_scale), shape[0]//2)
    pos = y_to_pos(y_b, l, shape, pos_block)

    cv2.rectangle(img, (pos_block[0]-12,pos_block[1]-6), (pos_block[0]+12,pos_block[1]+6), [235,240,245], -1)

    # lines
    for i in range(len(y_b)):
        cv2.line(img, pos[i], pos[i+1], [64,64,64], 2)
    # balls
    for i in range(len(y_b)):
        cv2.circle(img, pos[i+1], r[i], [int(255-255*(float(i)/len(y_b))),32,int(255*(float(i)/len(y_b)))], -1)
        cv2.circle(img, (pos[i+1][0]+r[i]//3,pos[i+1][1]-r[i]//3), r[i]//3, [235,240,245], -1)
        pos_last = pos
    
    return img

def solve_vec_interactive(method, f, t0, y0):
    t = time.time()
    h = t - t0
    y = method(f, t0, y0, h)
    frame = render_frame(y, l_scaled, m_scaled, frame_size, meter_scale=meter_scale)
    return frame, t, y


N = 1
g = 9.80665
l = np.array([1 for i in range(N)])
m = np.array([1 for i in range(N)])
k = 1
m_s = 1

def f(t, y):
	global mode_stop
	values = np.zeros_like(y)

	values[0,0] = y[0,1]
	values[0,1] = m[0]*(g*np.sin(y[1,0])*np.sin(y[1,0] + 1.5707963267949) - l[0]*y[1,1]**2*np.cos(y[1,0] + 1.5707963267949))/(m[0]*np.cos(y[1,0] + 1.5707963267949)**2 + m_s*np.sin(y[1,0] + 1.5707963267949)**2 + m_s*np.cos(y[1,0] + 1.5707963267949)**2)
	values[1,0] = y[1,1]
	values[1,1] = (g*(m[0] + m_s)*np.sin(y[1,0]) - 0.5*l[0]*m[0]*y[1,1]**2*np.sin(2.0*y[1,0] + 3.1415926535898))/(l[0]*(m[0]*np.sin(y[1,0] + 1.5707963267949)**2 - m[0] - m_s))
	return values



frame_size = 600
meter_scale = ((0.4 * frame_size)/np.sum(l))
l_scaled = l * meter_scale
m_scaled = mass_to_radius(m, 6, 12)


cv2.namedWindow('image')
#cv2.setMouseCallback('image', mouse_paint)
def nothing(x):
    pass
cv2.createTrackbar('slider', 'image', 0, frame_size, nothing)
mode_stop = False


t0 = time.time()
y0 = np.array([[-1,0], [1.57,0]])

last_pos = frame_size//2 + y0[0,0]*meter_scale
while(1):
	if mode_stop:
		h = time.time() - t0
		pos = cv2.getTrackbarPos('slider','image')
		diff = (pos-last_pos) / meter_scale
		y0[0,0] += diff
		y0[0,1] = 0#diff/h
		frame, t0, y0 = solve_vec_interactive(RK4_method, f, t0, y0)
		last_pos = pos
	else:
		frame, t0, y0 = solve_vec_interactive(RK4_method, f, t0, y0)
		pos = frame_size//2 + y0[0,0]*meter_scale
		cv2.setTrackbarPos('slider', 'image', int(pos))
		last_pos = pos
	#if mousedown:
	#	y0[0,0] = y0[0,0]+float(bx-lx)/(meter_scale)
	#	y0[0,1] = 0
	#print y0[0,]
	cv2.imshow('image', frame)


	k = cv2.waitKey(1) & 0xFF

	if k == ord('m'):
		mode_stop = not mode_stop
	elif k == 27:
		break

cv2.destroyAllWindows()
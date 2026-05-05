import sys
import socket
import rclpy
import queue
import signal
from rclpy.node import Node

from sensor_msgs.msg import Image

from flask import Flask, request, jsonify
import numpy as np
import cv2
from waitress import serve
from datetime import datetime
import threading

try:
    from cv_bridge import CvBridge, CvBridgeError
    _CV_BRIDGE_OK = True
except Exception:
    _CV_BRIDGE_OK = False


class ImagePub(Node):
    def __init__(self,nodeName,isTest,path):
        super().__init__(nodeName)
        self.publisher = self.create_publisher(Image, 'techman_image', 10)
        self.con = threading.Condition()
        self.imageQ = queue.Queue()
        self.leaveThread = False
        self.latest_frame = None  # most recent decoded frame for /api/snapshot
        if(isTest):
            self.t = threading.Thread(target = self.pub_data_thread, args=(False,))
            timer_period = 1.0
            self.img = cv2.imread(path)
            self.tmr = self.create_timer(timer_period, self.publish_test_image)
        else:
            self.t = threading.Thread(target = self.pub_data_thread, args=(True,))
        self.t.start()
                          
    def set_image_and_notify_send(self, img):
        self.con.acquire()
        self.imageQ.put(img)
        self.con.notify()
        self.con.release()
    def signal_handler(self,signal, frame):
        self.close_thread()
        sys.exit(0)
        
    def publish_test_image(self):
        self.img = cv2.flip(self.img, 1)
        self.set_image_and_notify_send(self.img)

    def image_publisher(self, image):
        if not _CV_BRIDGE_OK:
            return
        bridge = CvBridge()
        msg = bridge.cv2_to_imgmsg(image)
        self.get_logger().info('Publishing something !, queue size is ' + str(self.imageQ.qsize()))
        self.publisher.publish(msg)
    
    def close_thread(self):
        self.leaveThread = True
        self.con.acquire()
        self.con.notify()
        self.con.release()
        
    def pub_data_thread(self, isRequestData):
        self.con.acquire()
        while(True):
            self.con.wait()
            # Drain queue while holding lock, then release before slow work
            items = []
            while not self.imageQ.empty():
                items.append(self.imageQ.get())
            leave = self.leaveThread
            self.con.release()
            for raw in items:
                if isRequestData:
                    file2np = np.frombuffer(raw, np.uint8)
                    img = cv2.imdecode(file2np, cv2.IMREAD_UNCHANGED)
                else:
                    img = raw
                if img is not None:
                    self.latest_frame = img
                    self.image_publisher(img)
            if leave:
                return
            self.con.acquire()

    def snapshot(self):
        if self.latest_frame is None:
            return jsonify({"message": "no frame received yet"}), 503
        ret, buf = cv2.imencode(".jpg", self.latest_frame)
        if not ret:
            return jsonify({"message": "encode error"}), 500
        from flask import Response
        return Response(buf.tobytes(), mimetype="image/jpeg")

    def fake_result(self,m_method):
        # clsssification
        if m_method == 'CLS':
            # inference img here
            result = {
                "message": "success",
                "result": "NG", 
                "score": 0.987
            }
        # detection
        elif m_method == 'DET':            
            # inference img here                                    
            result = {
                "message":"success",
                "annotations": 
                [
                    { 
                        "box_cx": 150,
                        "box_cy": 150,
                        "box_w": 100,
                        "box_h": 100,                    
                        "label": "apple",                    
                        "score": 0.964,
                        "rotate": -45
                    },
                    { 
                        "box_cx": 550,
                        "box_cy": 550,
                        "box_w": 100,
                        "box_h": 100,
                        "label": "car",
                        "score": 1.000,
                        "rotation": 0
                    },
                    { 
                        "box_cx": 350,
                        "box_cy": 350,
                        "box_w": 150,
                        "box_h": 150,
                        "label": "mobilephone",
                        "score": 0.886,
                        "rotation": 135
                    }
                ],
                "result": None
            }
        # no method
        else:
            result = {            
                "message": "no method",
                "result": None            
            }
        return result

    def get_none(self):    
        print('\n[{0}] [{1}] -> Get()'.format(request.environ['REMOTE_ADDR'], datetime.now()))
        # user defined method
        result = {
            "result": "api",
            "message": "running",
        } 
        return jsonify(result)

    def get(self,m_method):
        print('\n[{0}] [{1}] -> Get({2})'.format(request.environ['REMOTE_ADDR'], datetime.now(), m_method))
        # user defined method
        if m_method == 'status':
            result = {
                "result": "status",
                "message": "im ok"
            }
        else:
            result = {
                "result": "fail",
                "message": "wrong request"            
            }
        return jsonify(result)

    def post(self,m_method):      
        print('\n[{0}] [{1}] -> Post({2})'.format(request.environ['REMOTE_ADDR'], datetime.now(), m_method))          
        # get key/value
        model_id = request.args.get('model_id')
        print('model_id: {}'.format(model_id))

        # convert image data
        self.set_image_and_notify_send(request.files['file'].read())

        result = self.fake_result(m_method)    

        return jsonify(result)
      
def set_route(app,node):
    app.route('/api/<string:m_method>', methods=['POST'])(node.post)
    app.route('/api/<string:m_method>', methods=['GET'])(node.get)
    app.route('/api', methods=['GET'])(node.get_none)
    app.route('/api/snapshot', methods=['GET'])(node.snapshot)

def main():
    rclpy.init(args=None)
    isTest = False
    app = Flask(__name__)
    if(isTest):
        try:
            print(sys.argv[1:])
        except :
            print("arg is not correct!")
            return
        
        node = ImagePub('image_pub',isTest,sys.argv[1])
    else:
        node = ImagePub('image_pub',isTest,None)

        set_route(app,node)
        print("Listening on an ip port:6189 combination")
        serve(app, port=6189)
    signal.signal(signal.SIGINT, node.signal_handler)
    
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

from dataclasses import dataclass
import numpy as np
court_scale=40

@dataclass
class PadelCourt:
    # REF: https://www.lta.org.uk/4ad2a4/siteassets/play/padel/file/lta-padel-court-guidance.pdf
    width: float = 10.0*court_scale
    length: float = 20.0*court_scale
    serve_line_from_back_line: float = 2.0*court_scale
    line_width:float = 0.05

    @classmethod
    @property
    def center_line(cls)->np.array:
        return np.array([
            (cls.width/2, cls.length),
            (cls.width/2, 0)
        ],dtype=np.int32).reshape(-1,1,2)
    @classmethod
    @property
    def net_line(cls)->np.array:
        return np.array([
            (0, cls.length/2),
            (cls.width, cls.length/2)
            ], dtype=np.int64
        ).reshape(-1, 1,2)
    @classmethod
    @property
    def near_serve_line(cls):
        return np.array([
            (0, cls.length-cls.serve_line_from_back_line),
            (cls.width, cls.length-cls.serve_line_from_back_line)

        ], np.int32).reshape(-1, 1, 2)

    @classmethod
    @property
    def far_serve_line(cls):
        return np.array([
            (0, cls.serve_line_from_back_line),
            (cls.width, cls.serve_line_from_back_line)
        ], dtype=np.int32).reshape(-1,1,2)    

    @classmethod
    @property
    def front_left(cls):
        return (0.0, cls.length)
    
    @classmethod
    @property
    def front_right(cls):
        return (cls.width, cls.length)

    @classmethod
    @property
    def back_left(cls):
        return (0.0, 0.0)

    @classmethod
    @property
    def back_right(cls):
        return ( cls.width,0.0)

    @classmethod
    @property
    def left_near_serve_line(cls):
        return (0.0, cls.length-cls.serve_line_from_back_line)
    

    @classmethod
    @property
    def right_near_serve_line(cls):
        return (cls.width, cls.length-cls.serve_line_from_back_line)

    @classmethod
    @property
    def left_far_serve_line(cls):
        return (0.0, cls.serve_line_from_back_line)

    @classmethod
    @property
    def right_far_serve_line(cls):
        return (cls.width, cls.serve_line_from_back_line)
    
    # Normalised:
    @classmethod
    @property
    def front_left_n(cls):
        return (0.0, cls.length/cls.length)
    
    @classmethod
    @property
    def front_right_n(cls):
        return (cls.width/cls.width, cls.length/cls.length)

    @classmethod
    @property
    def back_left_n(cls):
        return (0.0, 0.0)

    @classmethod
    @property
    def back_right_n(cls):
        return ( cls.width/cls.width,0.0)

    @classmethod
    @property
    def left_near_serve_line_n(cls):
        return (0.0, (cls.length-cls.serve_line_from_back_line)/cls.length)
    

    @classmethod
    @property
    def right_near_serve_line_n(cls):
        return (cls.width/cls.width, (cls.length-cls.serve_line_from_back_line)/cls.length)

    @classmethod
    @property
    def left_far_serve_line_n(cls):
        return (0.0, (cls.serve_line_from_back_line)/cls.length)

    @classmethod
    @property
    def right_far_serve_line_n(cls):
        return (cls.width/cls.width, cls.serve_line_from_back_line/cls.length)

corners_world = {
    "front_left":PadelCourt.front_left,
    "front_right":PadelCourt.front_right,
    "back_left": PadelCourt.back_left,
    "back_right": PadelCourt.back_right,
    
    "left_near_serve_line":PadelCourt.left_near_serve_line,
    "right_near_serve_line":PadelCourt.right_near_serve_line,
    "left_far_serve_line":PadelCourt.left_far_serve_line,
    "right_far_serve_line":PadelCourt.right_far_serve_line,
}
corners_world_3d = {
    "front_left":(*PadelCourt.front_left, 0.0),
    "front_right":(*PadelCourt.front_right, 0.0),
    "back_left": (*PadelCourt.back_left,0.0),
    "back_right": (*PadelCourt.back_right, 0.0),
    
    "left_near_serve_line":(*PadelCourt.left_near_serve_line,0.0),
    "right_near_serve_line":(*PadelCourt.right_near_serve_line,0.0),
    "left_far_serve_line":(*PadelCourt.left_far_serve_line,0.0),
    "right_far_serve_line":(*PadelCourt.right_far_serve_line,0.0),
}
corners_world_n = {
    "front_left":PadelCourt.front_left_n,
    "front_right":PadelCourt.front_right_n,
    "back_left": PadelCourt.back_left_n,
    "back_right": PadelCourt.back_right_n,
    
    "left_near_serve_line":PadelCourt.left_near_serve_line_n,
    "right_near_serve_line":PadelCourt.right_near_serve_line_n,
    "left_far_serve_line":PadelCourt.left_far_serve_line_n,
    "right_far_serve_line":PadelCourt.right_far_serve_line_n,
}
corners_world_3d_n = {
    "front_left":(*PadelCourt.front_left_n,0.0),
    "front_right":(*PadelCourt.front_right_n,0.0),
    "back_left": (*PadelCourt.back_left_n,0.0),
    "back_right": (*PadelCourt.back_right_n,0.0),
    
    "left_near_serve_line":(*PadelCourt.left_near_serve_line_n,0.0),
    "right_near_serve_line":(*PadelCourt.right_near_serve_line_n,0.0),
    "left_far_serve_line":(*PadelCourt.left_far_serve_line_n,0.0),
    "right_far_serve_line":(*PadelCourt.right_far_serve_line_n,0.0),
}

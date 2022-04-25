# -*- coding: utf-8 -*-

"""
Created on Mon Dec  6 20:23:27 2021

@author: ViJay
"""

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from matplotlib.widgets import Slider, Button
import re
import array as arr



"""
Define joints categorized in to three parts and accumalte them in body

"""

forelimbs = [ 23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
vertebra = [0, 1, 20, 2, 3]
handlimbs = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [forelimbs, vertebra, handlimbs]




"""
  Skeleton  for NTU RGB-D
"""
class VisualizationSkeleton(object):
    def __init__(self, file, save_path=None, pause_step=0.05):
        
        """
        Offset for moving the full figure in each direction. 
        This is changed via slider information
        
        """
        
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        
        """
        Offset of all 25 joints by which each joint should vary in each direction.
        This is changed via slider information
        
        """
        self.offset_x_j = np.zeros(25)
        self.offset_y_j = np.zeros(25)
        self.offset_z_j = np.zeros(25)
        
        """
        SLiders for updating the offsets
        """
        self.sliders = []
        
        self.selected_joint = 0
        
        self.file = file
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        
        self.xyz  = self.coordinate_xyz(self.file, max_body=1)

        
        self._pause_step = pause_step

        self.data = np.transpose(self.xyz, (3, 1, 2, 0))

       

        
        self.data = self.Normalized_Data(self.data)

        self.x = self.data[0, :, :, 0]
        self.y = self.data[0, :, :, 1]
        self.z = self.data[0, :, :, 2]

    def Skeleton(self, file):
        """
        Read the file and convert it into proper structured data with frame information and how many bodies are present in each frame and information on 25 joints for bodies in that particular frame.
        """
        with open(file, 'r') as f:
            skeleton_sequence = {}
            skeleton_sequence['Frame_Number'] = int(f.readline())
            skeleton_sequence['Frame_Information'] = []
            for t in range(skeleton_sequence['Frame_Number']):
                frame_information = {}
                frame_information['Body_Number'] = int(f.readline())
                frame_information['Body_Information'] = []
                for a in range(frame_information['Body_Number']):
                    body_information = {}
                    body_information_coordinates = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isresisted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_information = {
                        b: float(c)
                        for b, c in zip(body_information_coordinates, f.readline().split())
                    }
                    body_information['Joint_Number'] = int(f.readline())
                    body_information['Joint_Information'] = []
                    for v in range(body_information['Joint_Number']):
                        joint_information_coordinates = [
                            'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                            'orientationW', 'orientationX', 'orientationY',
                            'orientationZ', 'trackingState'
                        ]
                        joint_information = {
                            p: float(q)
                            for p, q in zip(joint_information_coordinates, f.readline().split())
                        }
                        body_information['Joint_Information'].append(joint_information)
                    frame_information['Body_Information'].append(body_information)
                skeleton_sequence['Frame_Information'].append(frame_information)
        return skeleton_sequence

    
    def coordinate_xyz(self, file, max_body=2, num_joint=25):
        """Read the file.
        Create the structured data for 25 joints with each joint having 3d-coordinates value
        This should be performed for all the frames for that particular motion figure
        """
        seq_info = self.Skeleton(file)
        data = np.zeros((3, seq_info['Frame_Number'], num_joint, max_body))  # (3,frame_nums,25 2)
        
        # Iterate for each frame
        for n, f in enumerate(seq_info['Frame_Information']):
            
            # Get the bodis information of bodies present in each frame
            for m, b in enumerate(f['Body_Information']):
                
                # Joint information of each body
                for j, v in enumerate(b['Joint_Information']):
                    if m < max_body and j < num_joint:
                        data[:, n, j, m] = [v['x'], v['y'], v['z']]
                    else:
                        pass

        return data

    def Normalized_Data(self, data):
        """
        Normalize the joints data and centralize it properly.
        """
        center_joint = data[0, :, 0, :]

        center_jointx = np.mean(center_joint[:, 0])
        center_jointy = np.mean(center_joint[:, 1])
        center_jointz = np.mean(center_joint[:, 2])

        center = np.array([center_jointx, center_jointy, center_jointz])
        data = data - center

        return data


    def visual_skeleton(self, action_name):
        
        """
        Create Figure animation for all the frames in file and display the motion of joints for different bodies.
        """
        # Initalize the plot
        fig = plt.figure(figsize=(15, 10), dpi=80)

        ax = Axes3D(fig)
        
        """
        Initalize the sliders position
         
        """
        x_axamp = plt.axes([0.84, 0.8 - (0 * 0.05), 0.12, 0.02])
        y_axamp = plt.axes([0.84, 0.8 - (1 * 0.05), 0.12, 0.02])
        z_axamp = plt.axes([0.84, 0.8 - (2 * 0.05), 0.12, 0.02])
        joint_axamp = plt.axes([0.84, 0.8 - (3 * 0.05), 0.12, 0.02])
        x_joint_axamp = plt.axes([0.84, 0.8 - (4 * 0.05), 0.12, 0.02])
        y_joint_axamp = plt.axes([0.84, 0.8 - (5 * 0.05), 0.12, 0.02])
        z_joint_axamp = plt.axes([0.84, 0.8 - (6 * 0.05), 0.12, 0.02])
        
        """
        Append the Slider 
        """
        self.sliders.append(Slider(x_axamp, 'x', -3, 3, 0))
        self.sliders.append(Slider(y_axamp, 'y', -3, 3, 0))
        self.sliders.append(Slider(z_axamp, 'z', -3, 3, 0))
        
        """
        With the joint slider select the joint for which you want to change the offset 
        and change the value of coordinates accordingly.
        """
        self.sliders.append( Slider(joint_axamp, 'Joint', 1, 25, valinit=1, valstep=1, dragging=True))
        self.sliders.append(Slider(x_joint_axamp, 'X Joint', -3, 3, 0))
        self.sliders.append(Slider(y_joint_axamp, 'Y Joint', -3, 3, 0))
        self.sliders.append(Slider(z_joint_axamp, 'Z Joint', -3, 3, 0))

    
        def update(val):
            """
            Update the figure offset based on values of slider.
            """   
            self.offset_x = self.sliders[0].val
            self.offset_y = self.sliders[1].val
            self.offset_z = self.sliders[2].val
           
            

        def update_selected_joint(val):
            """
                 Update the selected joint for updating joint offset.
            """   
            self.selected_joint = int(val-1)

            

        def update_joint_ax(val):
            """
                Update the joint offset based on values of slider.
            """   
            self.offset_x_j[self.selected_joint] = self.sliders[4].val
            self.offset_y_j[self.selected_joint] = self.sliders[5].val
            self.offset_z_j[self.selected_joint] = self.sliders[6].val


        # show every frame 3d skeleton

        def animate(frame_idx):
            
            """
                Update the Joints information for the frame_idx
            """
            # Get the data of 25 joints in frame_idx and add offsets to them
            x_frame = np.add(np.add(self.x[frame_idx], self.offset_x), self.offset_x_j)
            y_frame = np.add(np.add(self.y[frame_idx], self.offset_y), self.offset_y_j)
            z_frame = np.add(np.add(self.z[frame_idx], self.offset_z), self.offset_z_j)

            ax.clear()
            
            ax.set_xlim3d([-1.5, 1.5])
            ax.set_ylim3d([-1.5, 1.5])
            ax.set_zlim3d([-1.5, 1.5])
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.set_title(action_name)
            
            """
             Plot lines for ojint connection among different body parts
            """
            for part in body:
                x_plot = x_frame[part]
                y_plot = y_frame[part]
                z_plot = z_frame[part]
                ax.plot(x_plot, z_plot, y_plot, color='k', marker='o', markerfacecolor='r')
               

            # ax.set_facecolor('none')
            plt.pause(self._pause_step)


            """
            Show sliders in each figure
            
            """
            for i in np.arange(3):
                # samp.on_changed(update_slider)
                self.sliders[i].on_changed(update)
            self.sliders[3].on_changed(update_selected_joint)
            self.sliders[4].on_changed(update_joint_ax)
            self.sliders[5].on_changed(update_joint_ax)
            self.sliders[6].on_changed(update_joint_ax)

            
        """
        Identify the frames in dataset
        """
        fs = len(self.data[0, :, :, 0])
        
        """
        Call animation function with no. of frames and data
        """
        ani = animation.FuncAnimation(fig, animate, fs,
                                      interval=1, repeat=True)
        
        """
        Save the animation as mp4 and show the plot
        """
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('standup_multiple.mp4', writer=writer)

        plt.show()


action_name = {
    1: 'Drink water', 
    2: 'Eat meal', 
    3: 'Brushing teeth',
    4: 'Brushing hair',
    5: 'Drop',
    6: 'Pickup',
    7: 'Throw',
    8: 'Sitting down',
    9: 'Standing up',
    10: 'Clapping',
    11: 'Reading',
    12: 'Writting',
    13: 'Tear up paper',
    14: 'Wear Jacket',
    15: 'Take off Jacket',
    16: 'Wear Shoe',
    17: 'Take off a shoe',
    18: 'Wear on glasses',
    19: 'Take off glasses',
    20: 'Put on a hat',
    21: 'Take off a hat',
    22: 'Cheer up',
    23: 'Hand waving',
    24: 'Kicking something',
    25: 'Reach into Pocket',
    26: 'hopping(one foot jumping)',
    27: 'Jump up',
    28: 'Make a Phone call/answer phone',
    29: 'playing with phone/tablet',
    30: 'Typing on a keyboard',
    31: 'pointing to something with finger',
    32: 'taking a selfie',
    33: 'check time(from watch',
    34: 'rub two hands together',
    35: 'nod head/bow',
    36: 'shake head',
    37: 'wipe face',
    38: 'salute',
    39: 'put the palms together',
    40: 'cross hands in front (say stop)',
    41: 'sneeze/cough',
    42: 'staggering',
    43: 'falling',
    44: 'touch head(headache)',
    45: 'touch chest(stomachache/heart pain',
    46: 'touch back(backche)',
    47: 'touch neck(neckache)',
    48: 'nausea or vomitting condition',
    49: 'use a fan(with hand or paper)/feeling warm',
    50: 'punching/slapping other person',
    51: 'Kicking other person',
    52: 'pushing other person',
    53: 'pat on back of other person',
    54: 'point finger at the other person',
    55: 'hugging other person',
    56: 'giving something to other person',
    57: 'touch other persons pocket',
    58: 'handshaking',
    59: 'walking towards each other',
    60: 'walking apart from each other'
}

"""
Get figure action code from the file name and identify the action name from the dictionary
"""

def figure_action(file_path):
    file_name = file_path.split('/')[-1]
    action = re.findall(r"[A]{1}[0-9]+", file_name)[0]
    action = action.replace('A', '')
    action = int(action)
    return action_name[action]


if __name__ == '__main__':
    file_path = "C:/Users/Desktop/cap_project/nturgb+d_skeletons/S001C003P001R001A006.skeleton"
    action_name = figure_action(file_path)
    sk = VisualizationSkeleton(file_path, save_path="C:/Users/Downloads/skeleton-action-recognition-master")
    sk.visual_skeleton(action_name)



#!/usr/bin/env python

import asyncio
import websockets
import json
from attrdict import AttrDict
import numpy as np
import time
from tensorflow import keras


actions = np.array(["Left_punch","kiss","no_action","Right_punch","Wakanda"])
augmentations = ["CW Rotate 15","CCW Rotate 15","Left","Right","Zoom In","Zoom Out"]
label_map = {label:idx for idx,label in enumerate(actions)}
label_map

model = keras.models.load_model('../models_output/transformer.h5')



def keypointsToNumPy(results):
    pose_landmarks = np.array([[l.x,l.y,l.z,l.visibility] for l in results.pose_landmarks.landmark])
    lefthand_landmarks = np.array([[l.x,l.y,l.z] for l in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks and len(results.left_hand_landmarks.landmark)>0 else np.zeros(21*3)
    righthand_landmarks = np.array([[l.x,l.y,l.z] for l in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks and len(results.right_hand_landmarks.landmark)>0 else np.zeros(21*3)
    landmarks = np.concatenate([pose_landmarks.flatten(),lefthand_landmarks.flatten(),righthand_landmarks.flatten()])
    return landmarks


async def echo(websocket):
    frames = []
    async for message in websocket:
        # print(message)
        try:
            y = json.loads(message)
            results = AttrDict({
                "pose_landmarks":{
                    "landmark":[]
                },
                "left_hand_landmarks":{
                    "landmark":[]
                },
                "right_hand_landmarks":{
                    "landmark":[]
                }
            })
            results["pose_landmarks"]["landmark"] = y["poseLandmarks"]
            if "leftHandLandmarks" in y:
                results["left_hand_landmarks"]["landmark"] = y["leftHandLandmarks"]
            if "rightHandLandmarks" in y:
                results["right_hand_landmarks"]["landmark"] = y["rightHandLandmarks"]
            keypoints = keypointsToNumPy(results)
            frames.append(keypoints)
            # just get last 6 frames
            frames = frames[-6:]
            if len(frames) == 6:
                a = np.array(frames)
                # convert it to proper input (1,30,258)
                print(np.expand_dims(frames,axis=0).shape)
                # get the first prediction as we only pass one into it
                r = model.predict(np.expand_dims(frames,axis=0))[0]
                # r[r<0.5] = 0
                # print(actions[np.argmax(r)])
                j = {
                    "action": actions[np.argmax(r)],
                    "accuracy": r.tolist()
                }
                # await websocket.send(actions[np.argmax(r)] + "," + str(r))
                await websocket.send(json.dumps(j, separators=(',', ':')))
        except Exception as e:
            print(e)
        # await websocket.send(message + "server")

async def main():
    async with websockets.serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())
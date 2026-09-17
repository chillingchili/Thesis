This is for understanding the contents of this repository and also for developer notes.

bp.py is a simple test for blazepose and initialization

build_manifest.py is used to create a manifest of all the videos that will have their keypoints extracted

extract_keypoints.py is used to extract the keypoints of those pointed files by the manifest

npyvis.py is used to visualize raw npy files

pose_visualizer.py is used to overlay the blasepose model over the raw video, done in real-time so it's a bit laggy

--- NOTES ---

for the raw npy, the ff are not smooth in terms of keypoint tracking:

- CoachA_Drive_003

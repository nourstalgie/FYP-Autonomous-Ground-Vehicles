import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nourstalgie/FYP-Autonomous-Ground-Vehicles/PerceptionCorner/ws/install/detectnet_ros'

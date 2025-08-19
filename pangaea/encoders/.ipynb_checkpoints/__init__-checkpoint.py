# In pangaea/encoders/__init__.py
# Comprehensive OpenCV patch before ANY imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from opencv_patch import apply_comprehensive_opencv_patch
 
apply_comprehensive_opencv_patch()
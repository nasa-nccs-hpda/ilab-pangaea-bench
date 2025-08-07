import sys
import types
 
def apply_comprehensive_opencv_patch():
    """Apply comprehensive OpenCV compatibility patch before cv2 import"""
    class DictValue:
        def __init__(self, *args, **kwargs):
            if args:
                if len(args) == 2:
                    self.name = args[0]
                    self.value = args[1]
                else:
                    self.name = str(args[0]) if args else ""
                    self.value = ""
            else:
                self.name = kwargs.get('name', '')
                self.value = kwargs.get('value', '')
        def __str__(self):
            return f"{self.name}: {self.value}"
        def __repr__(self):
            return self.__str__()
    # Only patch if cv2 hasn't been imported yet
    if 'cv2' not in sys.modules:
        # Let's try to import the real cv2 first and just patch what's missing
        try:
            import cv2
            # If cv2 imports successfully, just add missing DictValue
            if not hasattr(cv2.dnn, 'DictValue'):
                cv2.dnn.DictValue = DictValue
                print("Patched missing cv2.dnn.DictValue in existing OpenCV")
            return
        except (ImportError, AttributeError):
            pass
    # If cv2 import failed or doesn't exist, create comprehensive mock
    print("Creating comprehensive OpenCV mock...")
    # Create mock modules
    mock_cv2 = types.ModuleType('cv2')
    mock_dnn = types.ModuleType('cv2.dnn')
    mock_typing = types.ModuleType('cv2.typing')
    # Add DictValue
    mock_dnn.DictValue = DictValue
    mock_typing.LayerId = DictValue
    # Data type constants
    mock_cv2.CV_8U = 0
    mock_cv2.CV_8S = 1
    mock_cv2.CV_16U = 2
    mock_cv2.CV_16S = 3
    mock_cv2.CV_32S = 4
    mock_cv2.CV_32F = 5
    mock_cv2.CV_64F = 6
    mock_cv2.CV_16F = 7
    # Interpolation constants
    mock_cv2.INTER_NEAREST = 0
    mock_cv2.INTER_LINEAR = 1
    mock_cv2.INTER_CUBIC = 2
    mock_cv2.INTER_AREA = 3
    mock_cv2.INTER_LANCZOS4 = 4
    mock_cv2.INTER_LINEAR_EXACT = 5  # This was missing
    mock_cv2.INTER_NEAREST_EXACT = 6
    mock_cv2.INTER_MAX = 7
    mock_cv2.WARP_FILL_OUTLIERS = 8
    mock_cv2.WARP_INVERSE_MAP = 16
    # Border types
    mock_cv2.BORDER_CONSTANT = 0
    mock_cv2.BORDER_REPLICATE = 1
    mock_cv2.BORDER_REFLECT = 2
    mock_cv2.BORDER_WRAP = 3
    mock_cv2.BORDER_REFLECT_101 = 4
    mock_cv2.BORDER_TRANSPARENT = 5
    mock_cv2.BORDER_REFLECT101 = 4  # Alias
    mock_cv2.BORDER_DEFAULT = 4
    mock_cv2.BORDER_ISOLATED = 16
    # Color conversion constants
    mock_cv2.COLOR_BGR2BGRA = 0
    mock_cv2.COLOR_RGB2RGBA = 0
    mock_cv2.COLOR_BGRA2BGR = 1
    mock_cv2.COLOR_RGBA2RGB = 1
    mock_cv2.COLOR_BGR2RGBA = 2
    mock_cv2.COLOR_RGB2BGRA = 2
    mock_cv2.COLOR_RGBA2BGR = 3
    mock_cv2.COLOR_BGRA2RGB = 3
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.COLOR_RGB2BGR = 4
    mock_cv2.COLOR_BGRA2RGBA = 5
    mock_cv2.COLOR_RGBA2BGRA = 5
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.COLOR_RGB2GRAY = 7
    mock_cv2.COLOR_GRAY2BGR = 8
    mock_cv2.COLOR_GRAY2RGB = 8
    mock_cv2.COLOR_GRAY2BGRA = 9
    mock_cv2.COLOR_GRAY2RGBA = 9
    mock_cv2.COLOR_BGRA2GRAY = 10
    mock_cv2.COLOR_RGBA2GRAY = 11
    # Morphological operation constants
    mock_cv2.MORPH_ERODE = 0
    mock_cv2.MORPH_DILATE = 1
    mock_cv2.MORPH_OPEN = 2
    mock_cv2.MORPH_CLOSE = 3
    mock_cv2.MORPH_GRADIENT = 4
    mock_cv2.MORPH_TOPHAT = 5
    mock_cv2.MORPH_BLACKHAT = 6
    mock_cv2.MORPH_HITMISS = 7
    # Morphological shapes
    mock_cv2.MORPH_RECT = 0
    mock_cv2.MORPH_CROSS = 1
    mock_cv2.MORPH_ELLIPSE = 2
    # Threshold types
    mock_cv2.THRESH_BINARY = 0
    mock_cv2.THRESH_BINARY_INV = 1
    mock_cv2.THRESH_TRUNC = 2
    mock_cv2.THRESH_TOZERO = 3
    mock_cv2.THRESH_TOZERO_INV = 4
    mock_cv2.THRESH_MASK = 7
    mock_cv2.THRESH_OTSU = 8
    mock_cv2.THRESH_TRIANGLE = 16
    # Add modules to mock_cv2
    mock_cv2.dnn = mock_dnn
    mock_cv2.typing = mock_typing
    # Add some basic functions that might be needed (dummy implementations)
    def dummy_function(*args, **kwargs):
        raise NotImplementedError("OpenCV function not available in mock mode")
    def dummy_array_function(*args, **kwargs):
        # Return a dummy numpy array for functions that should return arrays
        import numpy as np
        return np.array([])
    # Basic image operations
    mock_cv2.imread = dummy_function
    mock_cv2.imwrite = dummy_function
    mock_cv2.resize = dummy_function
    mock_cv2.cvtColor = dummy_function
    mock_cv2.warpAffine = dummy_function
    mock_cv2.warpPerspective = dummy_function
    mock_cv2.getRotationMatrix2D = dummy_array_function
    mock_cv2.getAffineTransform = dummy_array_function
    mock_cv2.getPerspectiveTransform = dummy_array_function
    # Filtering operations
    mock_cv2.blur = dummy_function
    mock_cv2.GaussianBlur = dummy_function
    mock_cv2.medianBlur = dummy_function
    mock_cv2.bilateralFilter = dummy_function
    mock_cv2.filter2D = dummy_function
    # Morphological operations
    mock_cv2.erode = dummy_function
    mock_cv2.dilate = dummy_function
    mock_cv2.morphologyEx = dummy_function
    mock_cv2.getStructuringElement = dummy_array_function
    # Insert into sys.modules
    sys.modules['cv2'] = mock_cv2
    sys.modules['cv2.dnn'] = mock_dnn
    sys.modules['cv2.typing'] = mock_typing
    print("Applied comprehensive OpenCV compatibility patch with extended constants")
"""
OpenCV compatibility patch - must be imported before cv2
"""
import sys
import types

def patch_cv2_before_import():
    """Patch cv2.dnn module before OpenCV imports"""
    
    # Create a mock dnn module with DictValue
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
    
    # Create mock cv2 and cv2.dnn modules
    mock_cv2 = types.ModuleType('cv2')
    mock_dnn = types.ModuleType('cv2.dnn')
    mock_dnn.DictValue = DictValue
    mock_cv2.dnn = mock_dnn
    
    # Insert into sys.modules before any cv2 import
    if 'cv2' not in sys.modules:
        sys.modules['cv2'] = mock_cv2
        sys.modules['cv2.dnn'] = mock_dnn
        print("Applied preemptive OpenCV DNN compatibility patch")
        return True
    return False

# Apply the patch immediately when this module is imported
patch_cv2_before_import()
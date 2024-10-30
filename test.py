import os
import sys

package_path = r'C:\Users\tommy\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages'
sys.path.append(package_path)
print(os.listdir(package_path))

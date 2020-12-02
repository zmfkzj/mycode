from distutils.core import setup
import py2exe
from pathlib import Path
import os

modulePath = Path('C:\\Users\\qlwla\\anaconda3\\envs\\datum\\Lib\\site-packages')
datumPluginPath = Path('C:\\Users\\qlwla\\anaconda3\\envs\\datum\\Lib\\site-packages\\datumaro\\plugins')
datumUtilsPath = Path('C:\\Users\\qlwla\\anaconda3\\envs\\datum\\Lib\\site-packages\\datumaro\\util')
datumInclude = lambda IncPath: [os.path.splitext(str(f.relative_to(modulePath)))[0].replace('\\','.') for r, _, _ in os.walk(IncPath) for f in Path(r).glob('*.py') if (Path(r)/f).name!='__init__.py' if 'tf_detection_api' not in r]
datumPlugins = datumInclude(datumPluginPath)
datumUtil = datumInclude(datumUtilsPath)
setup(console=['DatasetTools.py'],
      options={'py2exe':{'includes':['lxml','lxml.etree', 'lxml._elementpath',
                                     *datumPlugins, 
                                    #  *datumPlugins[:19], 
                                    #  'tensorflow'
                                    #  *datumUtil,
    #                                 # 'pathlib',
    #                                 'tensorflow'
                                    ],
                        #  'bundle_files':3,
                        'compressed':False,
                        }
              },
      # zipfile=None,
      )

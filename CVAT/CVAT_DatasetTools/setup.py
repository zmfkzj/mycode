from distutils.core import setup
import py2exe

setup(windows=[{'script':'DatasetTools.py'}],
      options={'py2exe':{'includes':['lxml','lxml.etree', 'lxml._elementpath',
                                    #  'protobuf',
    #                                 # 'pathlib',
    #                                 'tensorflow'
            ]
        }
    }
)

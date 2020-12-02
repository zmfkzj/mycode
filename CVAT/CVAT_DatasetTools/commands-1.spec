# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['commands.py'],
             pathex=['D:\\mycode\\CVAT\\CVAT_DatasetTools'],
             binaries=[],
             datas=[('C:\\Users\\qlwla\\anaconda3\\envs\\datum\\Lib\\site-packages\\datumaro', 'datumaro')],
             hiddenimports=['json','lxml.etree','tensorflow','attr','cython','git','matplotlib','cv2','PIL','pycocotools','yaml','skimage','tensorboardX'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['datumaro'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='commands',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )

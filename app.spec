# -*- mode: python -*-

block_cipher = None


a = Analysis(['app.py'],
             pathex=['./'],
             binaries=[],
             datas=[('venv/lib/python3.6/site-packages/ttkthemes', 'ttkthemes'),('venv/lib/python3.6/site-packages/PIL', 'PIL'), ('./appicon.png', 'appicon.png'), ('./appicon.ico', 'appicon.ico')],
             hiddenimports=['ttkthemes', 'PIL'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
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
          Tree('./pretrained/', prefix='pretrained'),
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True)



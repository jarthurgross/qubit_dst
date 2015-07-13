from setuptools import setup

requires = [
            'numpy',
           ]

setup(name='qubit_dst',
      version='0.0',
      py_modules=['dst_povm_sampling'],
      install_requires=requires,
      # Workaround from
      # https://github.com/numpy/numpy/issues/2434#issuecomment-65252402
      setup_requires=['numpy'],
      packages=['qubit_dst'],
      package_dir={'qubit_dst': 'src/qubit_dst'},
     )
